// Code generated from Plan.g4 by ANTLR 4.9. DO NOT EDIT.

package parser // Plan

import "github.com/antlr/antlr4/runtime/Go/antlr"

// PlanListener is a complete listener for a parse tree produced by PlanParser.
type PlanListener interface {
	antlr.ParseTreeListener

	// EnterShift is called when entering the Shift production.
	EnterShift(c *ShiftContext)

	// EnterBitOr is called when entering the BitOr production.
	EnterBitOr(c *BitOrContext)

	// EnterAddSub is called when entering the AddSub production.
	EnterAddSub(c *AddSubContext)

	// EnterParens is called when entering the Parens production.
	EnterParens(c *ParensContext)

	// EnterRelational is called when entering the Relational production.
	EnterRelational(c *RelationalContext)

	// EnterString is called when entering the String production.
	EnterString(c *StringContext)

	// EnterTerm is called when entering the Term production.
	EnterTerm(c *TermContext)

	// EnterFloating is called when entering the Floating production.
	EnterFloating(c *FloatingContext)

	// EnterRange is called when entering the Range production.
	EnterRange(c *RangeContext)

	// EnterUnary is called when entering the Unary production.
	EnterUnary(c *UnaryContext)

	// EnterLogicalOr is called when entering the LogicalOr production.
	EnterLogicalOr(c *LogicalOrContext)

	// EnterInteger is called when entering the Integer production.
	EnterInteger(c *IntegerContext)

	// EnterMulDivMod is called when entering the MulDivMod production.
	EnterMulDivMod(c *MulDivModContext)

	// EnterIdentifier is called when entering the Identifier production.
	EnterIdentifier(c *IdentifierContext)

	// EnterBitXor is called when entering the BitXor production.
	EnterBitXor(c *BitXorContext)

	// EnterBitAnd is called when entering the BitAnd production.
	EnterBitAnd(c *BitAndContext)

	// EnterLogicalAnd is called when entering the LogicalAnd production.
	EnterLogicalAnd(c *LogicalAndContext)

	// EnterEquality is called when entering the Equality production.
	EnterEquality(c *EqualityContext)

	// EnterBoolean is called when entering the Boolean production.
	EnterBoolean(c *BooleanContext)

	// EnterPower is called when entering the Power production.
	EnterPower(c *PowerContext)

	// ExitShift is called when exiting the Shift production.
	ExitShift(c *ShiftContext)

	// ExitBitOr is called when exiting the BitOr production.
	ExitBitOr(c *BitOrContext)

	// ExitAddSub is called when exiting the AddSub production.
	ExitAddSub(c *AddSubContext)

	// ExitParens is called when exiting the Parens production.
	ExitParens(c *ParensContext)

	// ExitRelational is called when exiting the Relational production.
	ExitRelational(c *RelationalContext)

	// ExitString is called when exiting the String production.
	ExitString(c *StringContext)

	// ExitTerm is called when exiting the Term production.
	ExitTerm(c *TermContext)

	// ExitFloating is called when exiting the Floating production.
	ExitFloating(c *FloatingContext)

	// ExitRange is called when exiting the Range production.
	ExitRange(c *RangeContext)

	// ExitUnary is called when exiting the Unary production.
	ExitUnary(c *UnaryContext)

	// ExitLogicalOr is called when exiting the LogicalOr production.
	ExitLogicalOr(c *LogicalOrContext)

	// ExitInteger is called when exiting the Integer production.
	ExitInteger(c *IntegerContext)

	// ExitMulDivMod is called when exiting the MulDivMod production.
	ExitMulDivMod(c *MulDivModContext)

	// ExitIdentifier is called when exiting the Identifier production.
	ExitIdentifier(c *IdentifierContext)

	// ExitBitXor is called when exiting the BitXor production.
	ExitBitXor(c *BitXorContext)

	// ExitBitAnd is called when exiting the BitAnd production.
	ExitBitAnd(c *BitAndContext)

	// ExitLogicalAnd is called when exiting the LogicalAnd production.
	ExitLogicalAnd(c *LogicalAndContext)

	// ExitEquality is called when exiting the Equality production.
	ExitEquality(c *EqualityContext)

	// ExitBoolean is called when exiting the Boolean production.
	ExitBoolean(c *BooleanContext)

	// ExitPower is called when exiting the Power production.
	ExitPower(c *PowerContext)
}
