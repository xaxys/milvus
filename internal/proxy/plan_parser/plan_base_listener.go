// Code generated from Plan.g4 by ANTLR 4.9. DO NOT EDIT.

package parser // Plan

import "github.com/antlr/antlr4/runtime/Go/antlr"

// BasePlanListener is a complete listener for a parse tree produced by PlanParser.
type BasePlanListener struct{}

var _ PlanListener = &BasePlanListener{}

// VisitTerminal is called when a terminal node is visited.
func (s *BasePlanListener) VisitTerminal(node antlr.TerminalNode) {}

// VisitErrorNode is called when an error node is visited.
func (s *BasePlanListener) VisitErrorNode(node antlr.ErrorNode) {}

// EnterEveryRule is called when any rule is entered.
func (s *BasePlanListener) EnterEveryRule(ctx antlr.ParserRuleContext) {}

// ExitEveryRule is called when any rule is exited.
func (s *BasePlanListener) ExitEveryRule(ctx antlr.ParserRuleContext) {}

// EnterShift is called when production Shift is entered.
func (s *BasePlanListener) EnterShift(ctx *ShiftContext) {}

// ExitShift is called when production Shift is exited.
func (s *BasePlanListener) ExitShift(ctx *ShiftContext) {}

// EnterBitOr is called when production BitOr is entered.
func (s *BasePlanListener) EnterBitOr(ctx *BitOrContext) {}

// ExitBitOr is called when production BitOr is exited.
func (s *BasePlanListener) ExitBitOr(ctx *BitOrContext) {}

// EnterAddSub is called when production AddSub is entered.
func (s *BasePlanListener) EnterAddSub(ctx *AddSubContext) {}

// ExitAddSub is called when production AddSub is exited.
func (s *BasePlanListener) ExitAddSub(ctx *AddSubContext) {}

// EnterParens is called when production Parens is entered.
func (s *BasePlanListener) EnterParens(ctx *ParensContext) {}

// ExitParens is called when production Parens is exited.
func (s *BasePlanListener) ExitParens(ctx *ParensContext) {}

// EnterRelational is called when production Relational is entered.
func (s *BasePlanListener) EnterRelational(ctx *RelationalContext) {}

// ExitRelational is called when production Relational is exited.
func (s *BasePlanListener) ExitRelational(ctx *RelationalContext) {}

// EnterString is called when production String is entered.
func (s *BasePlanListener) EnterString(ctx *StringContext) {}

// ExitString is called when production String is exited.
func (s *BasePlanListener) ExitString(ctx *StringContext) {}

// EnterTerm is called when production Term is entered.
func (s *BasePlanListener) EnterTerm(ctx *TermContext) {}

// ExitTerm is called when production Term is exited.
func (s *BasePlanListener) ExitTerm(ctx *TermContext) {}

// EnterFloating is called when production Floating is entered.
func (s *BasePlanListener) EnterFloating(ctx *FloatingContext) {}

// ExitFloating is called when production Floating is exited.
func (s *BasePlanListener) ExitFloating(ctx *FloatingContext) {}

// EnterRange is called when production Range is entered.
func (s *BasePlanListener) EnterRange(ctx *RangeContext) {}

// ExitRange is called when production Range is exited.
func (s *BasePlanListener) ExitRange(ctx *RangeContext) {}

// EnterUnary is called when production Unary is entered.
func (s *BasePlanListener) EnterUnary(ctx *UnaryContext) {}

// ExitUnary is called when production Unary is exited.
func (s *BasePlanListener) ExitUnary(ctx *UnaryContext) {}

// EnterLogicalOr is called when production LogicalOr is entered.
func (s *BasePlanListener) EnterLogicalOr(ctx *LogicalOrContext) {}

// ExitLogicalOr is called when production LogicalOr is exited.
func (s *BasePlanListener) ExitLogicalOr(ctx *LogicalOrContext) {}

// EnterInteger is called when production Integer is entered.
func (s *BasePlanListener) EnterInteger(ctx *IntegerContext) {}

// ExitInteger is called when production Integer is exited.
func (s *BasePlanListener) ExitInteger(ctx *IntegerContext) {}

// EnterMulDivMod is called when production MulDivMod is entered.
func (s *BasePlanListener) EnterMulDivMod(ctx *MulDivModContext) {}

// ExitMulDivMod is called when production MulDivMod is exited.
func (s *BasePlanListener) ExitMulDivMod(ctx *MulDivModContext) {}

// EnterIdentifier is called when production Identifier is entered.
func (s *BasePlanListener) EnterIdentifier(ctx *IdentifierContext) {}

// ExitIdentifier is called when production Identifier is exited.
func (s *BasePlanListener) ExitIdentifier(ctx *IdentifierContext) {}

// EnterBitXor is called when production BitXor is entered.
func (s *BasePlanListener) EnterBitXor(ctx *BitXorContext) {}

// ExitBitXor is called when production BitXor is exited.
func (s *BasePlanListener) ExitBitXor(ctx *BitXorContext) {}

// EnterBitAnd is called when production BitAnd is entered.
func (s *BasePlanListener) EnterBitAnd(ctx *BitAndContext) {}

// ExitBitAnd is called when production BitAnd is exited.
func (s *BasePlanListener) ExitBitAnd(ctx *BitAndContext) {}

// EnterLogicalAnd is called when production LogicalAnd is entered.
func (s *BasePlanListener) EnterLogicalAnd(ctx *LogicalAndContext) {}

// ExitLogicalAnd is called when production LogicalAnd is exited.
func (s *BasePlanListener) ExitLogicalAnd(ctx *LogicalAndContext) {}

// EnterEquality is called when production Equality is entered.
func (s *BasePlanListener) EnterEquality(ctx *EqualityContext) {}

// ExitEquality is called when production Equality is exited.
func (s *BasePlanListener) ExitEquality(ctx *EqualityContext) {}

// EnterBoolean is called when production Boolean is entered.
func (s *BasePlanListener) EnterBoolean(ctx *BooleanContext) {}

// ExitBoolean is called when production Boolean is exited.
func (s *BasePlanListener) ExitBoolean(ctx *BooleanContext) {}

// EnterPower is called when production Power is entered.
func (s *BasePlanListener) EnterPower(ctx *PowerContext) {}

// ExitPower is called when production Power is exited.
func (s *BasePlanListener) ExitPower(ctx *PowerContext) {}
