package rewriter

import (
	"fmt"
	"math"
	"sort"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

type bound struct {
	value     *planpb.GenericValue
	inclusive bool
	isLower   bool
	exprIndex int
	template  string
}

func isSupportedScalarForRange(dt schemapb.DataType) bool {
	switch dt {
	case schemapb.DataType_Int8,
		schemapb.DataType_Int16,
		schemapb.DataType_Int32,
		schemapb.DataType_Int64,
		schemapb.DataType_Float,
		schemapb.DataType_Double,
		schemapb.DataType_VarChar:
		return true
	default:
		return false
	}
}

// resolveEffectiveType returns (dt, ok) where ok indicates this column is eligible for range optimization.
// Eligible when the column is a supported scalar, or an array whose element type is a supported scalar,
// or a JSON field with a nested path (type will be determined from literal values).
func resolveEffectiveType(col *planpb.ColumnInfo) (schemapb.DataType, bool) {
	if col == nil {
		return schemapb.DataType_None, false
	}
	dt := col.GetDataType()
	if isSupportedScalarForRange(dt) {
		return dt, true
	}
	if dt == schemapb.DataType_Array {
		et := col.GetElementType()
		if isSupportedScalarForRange(et) {
			return et, true
		}
	}
	// JSON fields with nested paths are eligible; the effective type will be
	// determined from the literal value in the comparison.
	if dt == schemapb.DataType_JSON && len(col.GetNestedPath()) > 0 {
		// Return a placeholder type; actual type checking happens in resolveJSONEffectiveType
		return schemapb.DataType_JSON, true
	}
	return schemapb.DataType_None, false
}

// resolveJSONEffectiveType returns the effective type for a JSON field based on the literal value.
// Returns (type, ok) where ok is false if the value is not suitable for range optimization.
// Numeric literals share the Double comparison group so int and float bounds
// can still be merged. cmpGeneric preserves their concrete literal kinds and
// compares int64 against float64 without lossy promotion.
func resolveJSONEffectiveType(v *planpb.GenericValue) (schemapb.DataType, bool) {
	if v == nil || v.GetVal() == nil {
		return schemapb.DataType_None, false
	}
	switch v.GetVal().(type) {
	case *planpb.GenericValue_Int64Val:
		return schemapb.DataType_Double, true
	case *planpb.GenericValue_FloatVal:
		if math.IsNaN(v.GetFloatVal()) {
			return schemapb.DataType_None, false
		}
		return schemapb.DataType_Double, true
	case *planpb.GenericValue_StringVal:
		return schemapb.DataType_VarChar, true
	case *planpb.GenericValue_BoolVal:
		// Boolean comparisons don't have meaningful ranges
		return schemapb.DataType_None, false
	default:
		return schemapb.DataType_None, false
	}
}

func valueMatchesType(dt schemapb.DataType, v *planpb.GenericValue) bool {
	if v == nil || v.GetVal() == nil {
		return false
	}
	switch dt {
	case schemapb.DataType_Int8,
		schemapb.DataType_Int16,
		schemapb.DataType_Int32,
		schemapb.DataType_Int64:
		_, ok := v.GetVal().(*planpb.GenericValue_Int64Val)
		return ok
	case schemapb.DataType_Float, schemapb.DataType_Double:
		// For float columns, accept both float and int literal values
		switch v.GetVal().(type) {
		case *planpb.GenericValue_FloatVal:
			return !math.IsNaN(v.GetFloatVal())
		case *planpb.GenericValue_Int64Val:
			return true
		default:
			return false
		}
	case schemapb.DataType_VarChar:
		_, ok := v.GetVal().(*planpb.GenericValue_StringVal)
		return ok
	case schemapb.DataType_JSON:
		// For JSON, check if we can determine a valid type from the value
		_, ok := resolveJSONEffectiveType(v)
		return ok
	default:
		return false
	}
}

func resolveBinaryRangeEffectiveType(col *planpb.ColumnInfo, lower, upper *planpb.GenericValue) (schemapb.DataType, bool) {
	effDt, ok := resolveEffectiveType(col)
	if !ok {
		return schemapb.DataType_None, false
	}

	if effDt != schemapb.DataType_JSON {
		if !valueMatchesType(effDt, lower) || !valueMatchesType(effDt, upper) {
			return schemapb.DataType_None, false
		}
		return effDt, true
	}

	lowerType, lowerOK := resolveJSONEffectiveType(lower)
	upperType, upperOK := resolveJSONEffectiveType(upper)
	if !lowerOK || !upperOK || lowerType != upperType {
		return schemapb.DataType_None, false
	}
	return lowerType, true
}

func (v *visitor) combineAndRangePredicates(parts []*planpb.Expr) []*planpb.Expr {
	type group struct {
		col    *planpb.ColumnInfo
		effDt  schemapb.DataType // effective type for comparison
		lowers []bound
		uppers []bound
	}
	groups := map[string]*group{}
	// exprs not eligible for range optimization
	others := []int{}
	isRangeOp := func(op planpb.OpType) bool {
		return op == planpb.OpType_GreaterThan || op == planpb.OpType_GreaterEqual ||
			op == planpb.OpType_LessThan || op == planpb.OpType_LessEqual
	}
	for idx, e := range parts {
		u := e.GetUnaryRangeExpr()
		if u == nil || !isRangeOp(u.GetOp()) || u.GetValue() == nil {
			others = append(others, idx)
			continue
		}
		col := u.GetColumnInfo()
		if col == nil {
			others = append(others, idx)
			continue
		}
		// Only optimize for supported types and matching value type
		effDt, ok := resolveEffectiveType(col)
		if !ok || !valueMatchesType(effDt, u.GetValue()) {
			others = append(others, idx)
			continue
		}
		// For JSON fields, determine the actual effective type from the literal value
		if effDt == schemapb.DataType_JSON {
			var typeOk bool
			effDt, typeOk = resolveJSONEffectiveType(u.GetValue())
			if !typeOk {
				others = append(others, idx)
				continue
			}
		}
		// Group by column + effective type (for JSON, type depends on literal)
		key := columnKey(col) + fmt.Sprintf("|%d", effDt)
		g, ok := groups[key]
		if !ok {
			g = &group{col: col, effDt: effDt}
			groups[key] = g
		}
		b := bound{
			value:     u.GetValue(),
			inclusive: u.GetOp() == planpb.OpType_GreaterEqual || u.GetOp() == planpb.OpType_LessEqual,
			isLower:   u.GetOp() == planpb.OpType_GreaterThan || u.GetOp() == planpb.OpType_GreaterEqual,
			exprIndex: idx,
			template:  u.GetTemplateVariableName(),
		}
		if b.isLower {
			g.lowers = append(g.lowers, b)
		} else {
			g.uppers = append(g.uppers, b)
		}
	}
	used := make([]bool, len(parts))
	out := make([]*planpb.Expr, 0, len(parts))
	for _, idx := range others {
		out = append(out, parts[idx])
		used[idx] = true
	}
	for _, g := range groups {
		if len(g.lowers)+len(g.uppers) == 1 {
			continue
		}
		// Use the effective type stored in the group
		var bestLower *bound
		var lowerTpls []string
		for i := range g.lowers {
			if bestLower == nil {
				b := g.lowers[i]
				bestLower = &b
				if b.template != "" {
					lowerTpls = []string{b.template}
				}
				continue
			}
			c := cmpGeneric(g.effDt, g.lowers[i].value, bestLower.value)
			if c > 0 {
				b := g.lowers[i]
				bestLower = &b
				if b.template != "" {
					lowerTpls = []string{b.template}
				} else {
					lowerTpls = nil
				}
			} else if c == 0 {
				if g.lowers[i].template != "" {
					lowerTpls = append(lowerTpls, g.lowers[i].template)
				}
				if !g.lowers[i].inclusive && bestLower.inclusive {
					b := g.lowers[i]
					bestLower = &b
				}
			}
		}
		if bestLower != nil {
			bestLower.template = mergeTemplateNames(lowerTpls...)
		}
		var bestUpper *bound
		var upperTpls []string
		for i := range g.uppers {
			if bestUpper == nil {
				b := g.uppers[i]
				bestUpper = &b
				if b.template != "" {
					upperTpls = []string{b.template}
				}
				continue
			}
			c := cmpGeneric(g.effDt, g.uppers[i].value, bestUpper.value)
			if c < 0 {
				b := g.uppers[i]
				bestUpper = &b
				if b.template != "" {
					upperTpls = []string{b.template}
				} else {
					upperTpls = nil
				}
			} else if c == 0 {
				if g.uppers[i].template != "" {
					upperTpls = append(upperTpls, g.uppers[i].template)
				}
				if !g.uppers[i].inclusive && bestUpper.inclusive {
					b := g.uppers[i]
					bestUpper = &b
				}
			}
		}
		if bestUpper != nil {
			bestUpper.template = mergeTemplateNames(upperTpls...)
		}
		if bestLower != nil && bestUpper != nil {
			// Check if the interval is valid (non-empty)
			c := cmpGeneric(g.effDt, bestLower.value, bestUpper.value)
			isEmpty := false
			if c > 0 {
				// lower > upper: always empty
				isEmpty = true
			} else if c == 0 {
				// lower == upper: only valid if both bounds are inclusive
				if !bestLower.inclusive || !bestUpper.inclusive {
					isEmpty = true
				}
			}

			if isEmpty && !canFoldPredicateToBoolConstant(g.col) {
				continue
			}

			for _, b := range g.lowers {
				used[b.exprIndex] = true
			}
			for _, b := range g.uppers {
				used[b.exprIndex] = true
			}

			if isEmpty {
				// Empty interval → constant false
				out = append(out, newAlwaysFalseExpr())
			} else {
				out = append(out, newBinaryRangeExpr(g.col, bestLower.inclusive, bestUpper.inclusive, bestLower.value, bestUpper.value, bestLower.template, bestUpper.template))
			}
		} else if bestLower != nil {
			for _, b := range g.lowers {
				used[b.exprIndex] = true
			}
			op := planpb.OpType_GreaterThan
			if bestLower.inclusive {
				op = planpb.OpType_GreaterEqual
			}
			out = append(out, newUnaryRangeExpr(g.col, op, bestLower.value, bestLower.template))
		} else if bestUpper != nil {
			for _, b := range g.uppers {
				used[b.exprIndex] = true
			}
			op := planpb.OpType_LessThan
			if bestUpper.inclusive {
				op = planpb.OpType_LessEqual
			}
			out = append(out, newUnaryRangeExpr(g.col, op, bestUpper.value, bestUpper.template))
		}
	}
	for i := range parts {
		if !used[i] {
			out = append(out, parts[i])
		}
	}
	return out
}

func (v *visitor) combineOrRangePredicates(parts []*planpb.Expr) []*planpb.Expr {
	type key struct {
		colKey  string
		isLower bool
		effDt   schemapb.DataType // effective type for JSON fields
	}
	type group struct {
		col      *planpb.ColumnInfo
		effDt    schemapb.DataType
		dirLower bool
		bounds   []bound
	}
	groups := map[key]*group{}
	others := []int{}
	isRangeOp := func(op planpb.OpType) bool {
		return op == planpb.OpType_GreaterThan || op == planpb.OpType_GreaterEqual ||
			op == planpb.OpType_LessThan || op == planpb.OpType_LessEqual
	}
	for idx, e := range parts {
		u := e.GetUnaryRangeExpr()
		if u == nil || !isRangeOp(u.GetOp()) || u.GetValue() == nil {
			others = append(others, idx)
			continue
		}
		col := u.GetColumnInfo()
		if col == nil {
			others = append(others, idx)
			continue
		}
		effDt, ok := resolveEffectiveType(col)
		if !ok || !valueMatchesType(effDt, u.GetValue()) {
			others = append(others, idx)
			continue
		}
		// For JSON fields, determine the actual effective type from the literal value
		if effDt == schemapb.DataType_JSON {
			var typeOk bool
			effDt, typeOk = resolveJSONEffectiveType(u.GetValue())
			if !typeOk {
				others = append(others, idx)
				continue
			}
		}
		isLower := u.GetOp() == planpb.OpType_GreaterThan || u.GetOp() == planpb.OpType_GreaterEqual
		k := key{colKey: columnKey(col), isLower: isLower, effDt: effDt}
		g, ok := groups[k]
		if !ok {
			g = &group{col: col, effDt: effDt, dirLower: isLower}
			groups[k] = g
		}
		g.bounds = append(g.bounds, bound{
			value:     u.GetValue(),
			inclusive: u.GetOp() == planpb.OpType_GreaterEqual || u.GetOp() == planpb.OpType_LessEqual,
			isLower:   isLower,
			exprIndex: idx,
			template:  u.GetTemplateVariableName(),
		})
	}
	used := make([]bool, len(parts))
	out := make([]*planpb.Expr, 0, len(parts))
	for _, idx := range others {
		out = append(out, parts[idx])
		used[idx] = true
	}
	for _, g := range groups {
		if len(g.bounds) <= 1 {
			continue
		}
		if g.dirLower {
			var best *bound
			var bestTpls []string
			for i := range g.bounds {
				if best == nil {
					b := g.bounds[i]
					best = &b
					if b.template != "" {
						bestTpls = []string{b.template}
					}
					continue
				}
				// Use the effective type stored in the group
				c := cmpGeneric(g.effDt, g.bounds[i].value, best.value)
				if c < 0 {
					b := g.bounds[i]
					best = &b
					if b.template != "" {
						bestTpls = []string{b.template}
					} else {
						bestTpls = nil
					}
				} else if c == 0 {
					if g.bounds[i].template != "" {
						bestTpls = append(bestTpls, g.bounds[i].template)
					}
					if g.bounds[i].inclusive && !best.inclusive {
						b := g.bounds[i]
						best = &b
					}
				}
			}
			for _, b := range g.bounds {
				used[b.exprIndex] = true
			}
			op := planpb.OpType_GreaterThan
			if best.inclusive {
				op = planpb.OpType_GreaterEqual
			}
			out = append(out, newUnaryRangeExpr(g.col, op, best.value, bestTpls...))
		} else {
			var best *bound
			var bestTpls []string
			for i := range g.bounds {
				if best == nil {
					b := g.bounds[i]
					best = &b
					if b.template != "" {
						bestTpls = []string{b.template}
					}
					continue
				}
				// Use the effective type stored in the group
				c := cmpGeneric(g.effDt, g.bounds[i].value, best.value)
				if c > 0 {
					b := g.bounds[i]
					best = &b
					if b.template != "" {
						bestTpls = []string{b.template}
					} else {
						bestTpls = nil
					}
				} else if c == 0 {
					if g.bounds[i].template != "" {
						bestTpls = append(bestTpls, g.bounds[i].template)
					}
					if g.bounds[i].inclusive && !best.inclusive {
						b := g.bounds[i]
						best = &b
					}
				}
			}
			for _, b := range g.bounds {
				used[b.exprIndex] = true
			}
			op := planpb.OpType_LessThan
			if best.inclusive {
				op = planpb.OpType_LessEqual
			}
			out = append(out, newUnaryRangeExpr(g.col, op, best.value, bestTpls...))
		}
	}
	for i := range parts {
		if !used[i] {
			out = append(out, parts[i])
		}
	}
	return out
}

// integerRangesAdjacent reports whether an interval ending at upper and the
// next interval starting at lower (upper < lower) leave no value between them.
// Only typed integer columns qualify: JSON numbers and float columns can hold
// values between two integer literals. Normalizing exclusive bounds to
// inclusive (x < u as x <= u-1, x > l as x >= l+1) only widens a gap, so the
// intervals are adjacent exactly when both bounds are inclusive and
// upper + 1 == lower. The merged interval reuses the input literals.
func integerRangesAdjacent(dt schemapb.DataType, upper *planpb.GenericValue, upperInc bool, lower *planpb.GenericValue, lowerInc bool) bool {
	switch dt {
	case schemapb.DataType_Int8,
		schemapb.DataType_Int16,
		schemapb.DataType_Int32,
		schemapb.DataType_Int64:
	default:
		return false
	}
	if !upperInc || !lowerInc {
		return false
	}
	u := upper.GetInt64Val()
	return u != math.MaxInt64 && u+1 == lower.GetInt64Val()
}

func newBinaryRangeExpr(col *planpb.ColumnInfo, lowerInclusive bool, upperInclusive bool, lower *planpb.GenericValue, upper *planpb.GenericValue, templates ...string) *planpb.Expr {
	var lowerTpl, upperTpl string
	if len(templates) > 0 {
		lowerTpl = templates[0]
	}
	if len(templates) > 1 {
		upperTpl = templates[1]
	}
	return &planpb.Expr{
		Expr: &planpb.Expr_BinaryRangeExpr{
			BinaryRangeExpr: &planpb.BinaryRangeExpr{
				ColumnInfo:                col,
				LowerInclusive:            lowerInclusive,
				UpperInclusive:            upperInclusive,
				LowerValue:                lower,
				UpperValue:                upper,
				LowerTemplateVariableName: lowerTpl,
				UpperTemplateVariableName: upperTpl,
			},
		},
	}
}

// compareInt64ToFloat64 compares an int64 and float64 without lossy integer
// promotion. Callers must reject NaN before relying on the result.
func compareInt64ToFloat64(lhs int64, rhs float64) int {
	if math.IsNaN(rhs) {
		return 0
	}
	const (
		int64Lower = -0x1p63
		int64Upper = 0x1p63
	)
	if rhs < int64Lower {
		return 1
	}
	if rhs >= int64Upper {
		return -1
	}

	rhsInteger := int64(rhs)
	if lhs < rhsInteger {
		return -1
	}
	if lhs > rhsInteger {
		return 1
	}

	rhsIntegerAsFloat := float64(rhsInteger)
	if rhs > rhsIntegerAsFloat {
		return -1
	}
	if rhs < rhsIntegerAsFloat {
		return 1
	}
	return 0
}

// CompareRangeValues compares two supported range literals exactly.
func CompareRangeValues(a, b *planpb.GenericValue) (int, bool) {
	aType, aOK := resolveJSONEffectiveType(a)
	bType, bOK := resolveJSONEffectiveType(b)
	if !aOK || !bOK || aType != bType {
		return 0, false
	}
	return cmpGeneric(aType, a, b), true
}

// -1 means a < b, 0 means a == b, 1 means a > b
func cmpGeneric(dt schemapb.DataType, a, b *planpb.GenericValue) int {
	switch dt {
	case schemapb.DataType_Int8,
		schemapb.DataType_Int16,
		schemapb.DataType_Int32,
		schemapb.DataType_Int64:
		ai, bi := a.GetInt64Val(), b.GetInt64Val()
		if ai < bi {
			return -1
		}
		if ai > bi {
			return 1
		}
		return 0
	case schemapb.DataType_Float, schemapb.DataType_Double:
		switch a.GetVal().(type) {
		case *planpb.GenericValue_Int64Val:
			switch b.GetVal().(type) {
			case *planpb.GenericValue_Int64Val:
				ai, bi := a.GetInt64Val(), b.GetInt64Val()
				if ai < bi {
					return -1
				}
				if ai > bi {
					return 1
				}
				return 0
			case *planpb.GenericValue_FloatVal:
				return compareInt64ToFloat64(a.GetInt64Val(), b.GetFloatVal())
			}
		case *planpb.GenericValue_FloatVal:
			switch b.GetVal().(type) {
			case *planpb.GenericValue_Int64Val:
				return -compareInt64ToFloat64(b.GetInt64Val(), a.GetFloatVal())
			case *planpb.GenericValue_FloatVal:
				af, bf := a.GetFloatVal(), b.GetFloatVal()
				if af < bf {
					return -1
				}
				if af > bf {
					return 1
				}
				return 0
			}
		}
		// Should not happen because callers gate supported literal kinds.
		return 0
	case schemapb.DataType_String,
		schemapb.DataType_VarChar:
		as, bs := a.GetStringVal(), b.GetStringVal()
		if as < bs {
			return -1
		}
		if as > bs {
			return 1
		}
		return 0
	default:
		// Unsupported types are not optimized; callers gate with resolveEffectiveType.
		return 0
	}
}

// combineAndBinaryRanges merges BinaryRangeExpr nodes with AND semantics (intersection).
// Also handles mixing BinaryRangeExpr with UnaryRangeExpr.
func (v *visitor) combineAndBinaryRanges(parts []*planpb.Expr) []*planpb.Expr {
	type interval struct {
		lower         *planpb.GenericValue
		lowerInc      bool
		lowerTemplate string
		upper         *planpb.GenericValue
		upperInc      bool
		upperTemplate string
		exprIndex     int
		isBinaryRange bool
	}
	type group struct {
		col       *planpb.ColumnInfo
		effDt     schemapb.DataType
		intervals []interval
	}
	groups := map[string]*group{}
	others := []int{}

	for idx, e := range parts {
		// Try BinaryRangeExpr
		if bre := e.GetBinaryRangeExpr(); bre != nil {
			col := bre.GetColumnInfo()
			if col == nil {
				others = append(others, idx)
				continue
			}
			effDt, ok := resolveBinaryRangeEffectiveType(col, bre.GetLowerValue(), bre.GetUpperValue())
			if !ok {
				others = append(others, idx)
				continue
			}
			key := columnKey(col) + fmt.Sprintf("|%d", effDt)
			g, exists := groups[key]
			if !exists {
				g = &group{col: col, effDt: effDt}
				groups[key] = g
			}
			g.intervals = append(g.intervals, interval{
				lower:         bre.GetLowerValue(),
				lowerInc:      bre.GetLowerInclusive(),
				lowerTemplate: bre.GetLowerTemplateVariableName(),
				upper:         bre.GetUpperValue(),
				upperInc:      bre.GetUpperInclusive(),
				upperTemplate: bre.GetUpperTemplateVariableName(),
				exprIndex:     idx,
				isBinaryRange: true,
			})
			continue
		}

		// Try UnaryRangeExpr (range ops only)
		if ure := e.GetUnaryRangeExpr(); ure != nil {
			op := ure.GetOp()
			if op == planpb.OpType_GreaterThan || op == planpb.OpType_GreaterEqual ||
				op == planpb.OpType_LessThan || op == planpb.OpType_LessEqual {
				col := ure.GetColumnInfo()
				if col == nil {
					others = append(others, idx)
					continue
				}
				effDt, ok := resolveEffectiveType(col)
				if !ok || !valueMatchesType(effDt, ure.GetValue()) {
					others = append(others, idx)
					continue
				}
				if effDt == schemapb.DataType_JSON {
					var typeOk bool
					effDt, typeOk = resolveJSONEffectiveType(ure.GetValue())
					if !typeOk {
						others = append(others, idx)
						continue
					}
				}
				key := columnKey(col) + fmt.Sprintf("|%d", effDt)
				g, exists := groups[key]
				if !exists {
					g = &group{col: col, effDt: effDt}
					groups[key] = g
				}
				isLower := op == planpb.OpType_GreaterThan || op == planpb.OpType_GreaterEqual
				inc := op == planpb.OpType_GreaterEqual || op == planpb.OpType_LessEqual
				if isLower {
					g.intervals = append(g.intervals, interval{
						lower:         ure.GetValue(),
						lowerInc:      inc,
						lowerTemplate: ure.GetTemplateVariableName(),
						upper:         nil,
						upperInc:      false,
						exprIndex:     idx,
						isBinaryRange: false,
					})
				} else {
					g.intervals = append(g.intervals, interval{
						lower:         nil,
						lowerInc:      false,
						upper:         ure.GetValue(),
						upperInc:      inc,
						upperTemplate: ure.GetTemplateVariableName(),
						exprIndex:     idx,
						isBinaryRange: false,
					})
				}
				continue
			}
		}

		// Not a range expr we can optimize
		others = append(others, idx)
	}

	used := make([]bool, len(parts))
	out := make([]*planpb.Expr, 0, len(parts))
	for _, idx := range others {
		out = append(out, parts[idx])
		used[idx] = true
	}

	for _, g := range groups {
		if len(g.intervals) == 0 {
			continue
		}
		if len(g.intervals) == 1 {
			// Single interval, keep as is
			continue
		}

		// Compute intersection: max lower, min upper
		var finalLower *planpb.GenericValue
		var finalLowerInc bool
		var finalLowerTpls []string
		var finalUpper *planpb.GenericValue
		var finalUpperInc bool
		var finalUpperTpls []string

		for _, iv := range g.intervals {
			if iv.lower != nil {
				if finalLower == nil {
					finalLower = iv.lower
					finalLowerInc = iv.lowerInc
					if iv.lowerTemplate != "" {
						finalLowerTpls = []string{iv.lowerTemplate}
					}
				} else {
					c := cmpGeneric(g.effDt, iv.lower, finalLower)
					if c > 0 {
						finalLower = iv.lower
						finalLowerInc = iv.lowerInc
						if iv.lowerTemplate != "" {
							finalLowerTpls = []string{iv.lowerTemplate}
						} else {
							finalLowerTpls = nil
						}
					} else if c == 0 {
						if iv.lowerTemplate != "" {
							finalLowerTpls = append(finalLowerTpls, iv.lowerTemplate)
						}
						if !iv.lowerInc {
							finalLower = iv.lower
							finalLowerInc = iv.lowerInc
						}
					}
				}
			}
			if iv.upper != nil {
				if finalUpper == nil {
					finalUpper = iv.upper
					finalUpperInc = iv.upperInc
					if iv.upperTemplate != "" {
						finalUpperTpls = []string{iv.upperTemplate}
					}
				} else {
					c := cmpGeneric(g.effDt, iv.upper, finalUpper)
					if c < 0 {
						finalUpper = iv.upper
						finalUpperInc = iv.upperInc
						if iv.upperTemplate != "" {
							finalUpperTpls = []string{iv.upperTemplate}
						} else {
							finalUpperTpls = nil
						}
					} else if c == 0 {
						if iv.upperTemplate != "" {
							finalUpperTpls = append(finalUpperTpls, iv.upperTemplate)
						}
						if !iv.upperInc {
							finalUpper = iv.upper
							finalUpperInc = iv.upperInc
						}
					}
				}
			}
		}
		finalLowerTpl := mergeTemplateNames(finalLowerTpls...)
		finalUpperTpl := mergeTemplateNames(finalUpperTpls...)

		// Check if intersection is empty
		if finalLower != nil && finalUpper != nil {
			c := cmpGeneric(g.effDt, finalLower, finalUpper)
			isEmpty := false
			if c > 0 {
				isEmpty = true
			} else if c == 0 {
				// Equal bounds: only valid if both inclusive
				if !finalLowerInc || !finalUpperInc {
					isEmpty = true
				}
			}
			if isEmpty {
				if !canFoldPredicateToBoolConstant(g.col) {
					continue
				}
				// Empty intersection → constant false
				for _, iv := range g.intervals {
					used[iv.exprIndex] = true
				}
				out = append(out, newAlwaysFalseExpr())
				continue
			}
		}

		// Mark all intervals as used
		for _, iv := range g.intervals {
			used[iv.exprIndex] = true
		}

		// Emit the merged interval
		if finalLower != nil && finalUpper != nil {
			out = append(out, newBinaryRangeExpr(g.col, finalLowerInc, finalUpperInc, finalLower, finalUpper, finalLowerTpl, finalUpperTpl))
		} else if finalLower != nil {
			op := planpb.OpType_GreaterThan
			if finalLowerInc {
				op = planpb.OpType_GreaterEqual
			}
			out = append(out, newUnaryRangeExpr(g.col, op, finalLower, finalLowerTpl))
		} else if finalUpper != nil {
			op := planpb.OpType_LessThan
			if finalUpperInc {
				op = planpb.OpType_LessEqual
			}
			out = append(out, newUnaryRangeExpr(g.col, op, finalUpper, finalUpperTpl))
		}
	}

	// Add unused parts
	for i := range parts {
		if !used[i] {
			out = append(out, parts[i])
		}
	}

	return out
}

// combineOrBinaryRanges merges BinaryRangeExpr nodes with OR semantics (union if overlapping/adjacent;
// for integer columns [a, v] and [v+1, b] are adjacent).
// Also handles mixing BinaryRangeExpr with UnaryRangeExpr.
func (v *visitor) combineOrBinaryRanges(parts []*planpb.Expr) []*planpb.Expr {
	type interval struct {
		lower         *planpb.GenericValue
		lowerInc      bool
		lowerTemplate string
		upper         *planpb.GenericValue
		upperInc      bool
		upperTemplate string
		exprIndex     int
		isBinaryRange bool
	}
	type group struct {
		col       *planpb.ColumnInfo
		effDt     schemapb.DataType
		intervals []interval
	}
	groups := map[string]*group{}
	others := []int{}

	for idx, e := range parts {
		// Try BinaryRangeExpr
		if bre := e.GetBinaryRangeExpr(); bre != nil {
			col := bre.GetColumnInfo()
			if col == nil {
				others = append(others, idx)
				continue
			}
			effDt, ok := resolveBinaryRangeEffectiveType(col, bre.GetLowerValue(), bre.GetUpperValue())
			if !ok {
				others = append(others, idx)
				continue
			}
			key := columnKey(col) + fmt.Sprintf("|%d", effDt)
			g, exists := groups[key]
			if !exists {
				g = &group{col: col, effDt: effDt}
				groups[key] = g
			}
			g.intervals = append(g.intervals, interval{
				lower:         bre.GetLowerValue(),
				lowerInc:      bre.GetLowerInclusive(),
				lowerTemplate: bre.GetLowerTemplateVariableName(),
				upper:         bre.GetUpperValue(),
				upperInc:      bre.GetUpperInclusive(),
				upperTemplate: bre.GetUpperTemplateVariableName(),
				exprIndex:     idx,
				isBinaryRange: true,
			})
			continue
		}

		// Try UnaryRangeExpr
		if ure := e.GetUnaryRangeExpr(); ure != nil {
			op := ure.GetOp()
			if op == planpb.OpType_GreaterThan || op == planpb.OpType_GreaterEqual ||
				op == planpb.OpType_LessThan || op == planpb.OpType_LessEqual {
				col := ure.GetColumnInfo()
				if col == nil {
					others = append(others, idx)
					continue
				}
				effDt, ok := resolveEffectiveType(col)
				if !ok || !valueMatchesType(effDt, ure.GetValue()) {
					others = append(others, idx)
					continue
				}
				if effDt == schemapb.DataType_JSON {
					var typeOk bool
					effDt, typeOk = resolveJSONEffectiveType(ure.GetValue())
					if !typeOk {
						others = append(others, idx)
						continue
					}
				}
				key := columnKey(col) + fmt.Sprintf("|%d", effDt)
				g, exists := groups[key]
				if !exists {
					g = &group{col: col, effDt: effDt}
					groups[key] = g
				}
				isLower := op == planpb.OpType_GreaterThan || op == planpb.OpType_GreaterEqual
				inc := op == planpb.OpType_GreaterEqual || op == planpb.OpType_LessEqual
				if isLower {
					g.intervals = append(g.intervals, interval{
						lower:         ure.GetValue(),
						lowerInc:      inc,
						lowerTemplate: ure.GetTemplateVariableName(),
						upper:         nil,
						upperInc:      false,
						exprIndex:     idx,
						isBinaryRange: false,
					})
				} else {
					g.intervals = append(g.intervals, interval{
						lower:         nil,
						lowerInc:      false,
						upper:         ure.GetValue(),
						upperInc:      inc,
						upperTemplate: ure.GetTemplateVariableName(),
						exprIndex:     idx,
						isBinaryRange: false,
					})
				}
				continue
			}
		}

		others = append(others, idx)
	}

	used := make([]bool, len(parts))
	out := make([]*planpb.Expr, 0, len(parts))
	for _, idx := range others {
		out = append(out, parts[idx])
		used[idx] = true
	}

	for _, g := range groups {
		if len(g.intervals) == 0 {
			continue
		}
		if len(g.intervals) == 1 {
			// Single interval, keep as is
			continue
		}

		// Keep unbounded, non-finite, and empty/invalid intervals unchanged.
		// In particular, dropping an empty nullable interval could lose UNKNOWN
		// under an outer NOT. Only union ordinary bounded intervals here.
		canMerge := true
		for _, iv := range g.intervals {
			if iv.lower == nil || iv.upper == nil {
				canMerge = false
				break
			}
			for _, value := range []*planpb.GenericValue{iv.lower, iv.upper} {
				if floatValue, ok := value.GetVal().(*planpb.GenericValue_FloatVal); ok &&
					(math.IsNaN(floatValue.FloatVal) || math.IsInf(floatValue.FloatVal, 0)) {
					canMerge = false
				}
			}
			c := cmpGeneric(g.effDt, iv.lower, iv.upper)
			if c > 0 || (c == 0 && (!iv.lowerInc || !iv.upperInc)) {
				canMerge = false
			}
			if !canMerge {
				break
			}
		}
		if !canMerge {
			continue
		}

		// One-shot flattening exposes the whole OR chain. Sort once and sweep
		// all intervals in O(N log N), retaining gaps and endpoint inclusivity.
		sort.Slice(g.intervals, func(i, j int) bool {
			c := cmpGeneric(g.effDt, g.intervals[i].lower, g.intervals[j].lower)
			return c < 0 || (c == 0 && g.intervals[i].lowerInc && !g.intervals[j].lowerInc)
		})
		type mergedInterval struct {
			interval
			lowerTpls []string
			upperTpls []string
		}
		merged := make([]mergedInterval, 0, len(g.intervals))
		for _, iv := range g.intervals {
			var lt, ut []string
			if iv.lowerTemplate != "" {
				lt = []string{iv.lowerTemplate}
			}
			if iv.upperTemplate != "" {
				ut = []string{iv.upperTemplate}
			}
			if len(merged) == 0 {
				merged = append(merged, mergedInterval{interval: iv, lowerTpls: lt, upperTpls: ut})
				continue
			}
			last := &merged[len(merged)-1]
			c := cmpGeneric(g.effDt, last.upper, iv.lower)
			if (c < 0 && !integerRangesAdjacent(g.effDt, last.upper, last.upperInc, iv.lower, iv.lowerInc)) ||
				(c == 0 && !last.upperInc && !iv.lowerInc) {
				merged = append(merged, mergedInterval{interval: iv, lowerTpls: lt, upperTpls: ut})
				continue
			}
			if cmpGeneric(g.effDt, last.lower, iv.lower) == 0 {
				last.lowerInc = last.lowerInc || iv.lowerInc
				if iv.lowerTemplate != "" {
					last.lowerTpls = append(last.lowerTpls, iv.lowerTemplate)
				}
			}
			c = cmpGeneric(g.effDt, last.upper, iv.upper)
			if c < 0 {
				last.upper = iv.upper
				last.upperInc = iv.upperInc
				last.upperTpls = ut
			} else if c == 0 {
				last.upperInc = last.upperInc || iv.upperInc
				if iv.upperTemplate != "" {
					last.upperTpls = append(last.upperTpls, iv.upperTemplate)
				}
			}
			last.exprIndex = -1 // This cluster needs a new merged predicate.
		}
		if len(merged) == len(g.intervals) {
			continue
		}
		for _, iv := range g.intervals {
			used[iv.exprIndex] = true
		}
		for _, iv := range merged {
			if iv.exprIndex >= 0 {
				out = append(out, parts[iv.exprIndex])
			} else {
				out = append(out, newBinaryRangeExpr(
					g.col,
					iv.lowerInc,
					iv.upperInc,
					iv.lower,
					iv.upper,
					mergeTemplateNames(iv.lowerTpls...),
					mergeTemplateNames(iv.upperTpls...),
				))
			}
		}
	}

	// Add unused parts
	for i := range parts {
		if !used[i] {
			out = append(out, parts[i])
		}
	}

	return out
}
