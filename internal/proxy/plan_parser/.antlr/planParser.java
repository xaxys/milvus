// Generated from /home/xa/milvus/internal/proxy/plan_parser/plan.g4 by ANTLR 4.8
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.misc.*;
import org.antlr.v4.runtime.tree.*;
import java.util.List;
import java.util.Iterator;
import java.util.ArrayList;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast"})
public class planParser extends Parser {
	static { RuntimeMetaData.checkVersion("4.8", RuntimeMetaData.VERSION); }

	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		T__0=1, T__1=2, T__2=3, T__3=4, T__4=5, BOOL=6, INT8=7, INT16=8, INT32=9, 
		INT64=10, FLOAT=11, DOUBLE=12, LT=13, LE=14, GT=15, GE=16, EQ=17, NE=18, 
		ADD=19, SUB=20, MUL=21, DIV=22, MOD=23, SHL=24, SHR=25, BAND=26, BOR=27, 
		BXOR=28, AND=29, OR=30, BNOT=31, NOT=32, IN=33, TRUE=34, FALSE=35, Identifier=36, 
		StringLiteral=37, IntegerConstant=38, FloatingConstant=39, BooleanConstant=40, 
		Whitespace=41, Newline=42;
	public static final int
		RULE_constExpr = 0, RULE_expr = 1, RULE_termList = 2, RULE_typeName = 3;
	private static String[] makeRuleNames() {
		return new String[] {
			"constExpr", "expr", "termList", "typeName"
		};
	}
	public static final String[] ruleNames = makeRuleNames();

	private static String[] makeLiteralNames() {
		return new String[] {
			null, "'('", "')'", "'['", "','", "']'", "'bool'", "'int8'", "'int16'", 
			"'int32'", "'int64'", "'float'", "'double'", "'<'", "'<='", "'>'", "'>='", 
			"'=='", "'!='", "'+'", "'-'", "'*'", "'/'", "'%'", "'<<'", "'>>'", "'&'", 
			"'|'", "'^'", "'&&'", "'||'", "'~'", "'!'", "'in'", "'true'", "'false'"
		};
	}
	private static final String[] _LITERAL_NAMES = makeLiteralNames();
	private static String[] makeSymbolicNames() {
		return new String[] {
			null, null, null, null, null, null, "BOOL", "INT8", "INT16", "INT32", 
			"INT64", "FLOAT", "DOUBLE", "LT", "LE", "GT", "GE", "EQ", "NE", "ADD", 
			"SUB", "MUL", "DIV", "MOD", "SHL", "SHR", "BAND", "BOR", "BXOR", "AND", 
			"OR", "BNOT", "NOT", "IN", "TRUE", "FALSE", "Identifier", "StringLiteral", 
			"IntegerConstant", "FloatingConstant", "BooleanConstant", "Whitespace", 
			"Newline"
		};
	}
	private static final String[] _SYMBOLIC_NAMES = makeSymbolicNames();
	public static final Vocabulary VOCABULARY = new VocabularyImpl(_LITERAL_NAMES, _SYMBOLIC_NAMES);

	/**
	 * @deprecated Use {@link #VOCABULARY} instead.
	 */
	@Deprecated
	public static final String[] tokenNames;
	static {
		tokenNames = new String[_SYMBOLIC_NAMES.length];
		for (int i = 0; i < tokenNames.length; i++) {
			tokenNames[i] = VOCABULARY.getLiteralName(i);
			if (tokenNames[i] == null) {
				tokenNames[i] = VOCABULARY.getSymbolicName(i);
			}

			if (tokenNames[i] == null) {
				tokenNames[i] = "<INVALID>";
			}
		}
	}

	@Override
	@Deprecated
	public String[] getTokenNames() {
		return tokenNames;
	}

	@Override

	public Vocabulary getVocabulary() {
		return VOCABULARY;
	}

	@Override
	public String getGrammarFileName() { return "plan.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public ATN getATN() { return _ATN; }

	public planParser(TokenStream input) {
		super(input);
		_interp = new ParserATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	public static class ConstExprContext extends ParserRuleContext {
		public ConstExprContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_constExpr; }
	 
		public ConstExprContext() { }
		public void copyFrom(ConstExprContext ctx) {
			super.copyFrom(ctx);
		}
	}
	public static class ConstBitXorContext extends ConstExprContext {
		public List<ConstExprContext> constExpr() {
			return getRuleContexts(ConstExprContext.class);
		}
		public ConstExprContext constExpr(int i) {
			return getRuleContext(ConstExprContext.class,i);
		}
		public TerminalNode BXOR() { return getToken(planParser.BXOR, 0); }
		public ConstBitXorContext(ConstExprContext ctx) { copyFrom(ctx); }
	}
	public static class ConstEqualityContext extends ConstExprContext {
		public Token op;
		public List<ConstExprContext> constExpr() {
			return getRuleContexts(ConstExprContext.class);
		}
		public ConstExprContext constExpr(int i) {
			return getRuleContext(ConstExprContext.class,i);
		}
		public TerminalNode EQ() { return getToken(planParser.EQ, 0); }
		public TerminalNode NE() { return getToken(planParser.NE, 0); }
		public ConstEqualityContext(ConstExprContext ctx) { copyFrom(ctx); }
	}
	public static class ConstUnaryContext extends ConstExprContext {
		public Token op;
		public ConstExprContext constExpr() {
			return getRuleContext(ConstExprContext.class,0);
		}
		public TerminalNode ADD() { return getToken(planParser.ADD, 0); }
		public TerminalNode SUB() { return getToken(planParser.SUB, 0); }
		public TerminalNode BNOT() { return getToken(planParser.BNOT, 0); }
		public TerminalNode NOT() { return getToken(planParser.NOT, 0); }
		public ConstUnaryContext(ConstExprContext ctx) { copyFrom(ctx); }
	}
	public static class StringContext extends ConstExprContext {
		public TerminalNode StringLiteral() { return getToken(planParser.StringLiteral, 0); }
		public StringContext(ConstExprContext ctx) { copyFrom(ctx); }
	}
	public static class ConstRelationalContext extends ConstExprContext {
		public Token op;
		public List<ConstExprContext> constExpr() {
			return getRuleContexts(ConstExprContext.class);
		}
		public ConstExprContext constExpr(int i) {
			return getRuleContext(ConstExprContext.class,i);
		}
		public TerminalNode LT() { return getToken(planParser.LT, 0); }
		public TerminalNode LE() { return getToken(planParser.LE, 0); }
		public TerminalNode GT() { return getToken(planParser.GT, 0); }
		public TerminalNode GE() { return getToken(planParser.GE, 0); }
		public ConstRelationalContext(ConstExprContext ctx) { copyFrom(ctx); }
	}
	public static class FloatingContext extends ConstExprContext {
		public TerminalNode FloatingConstant() { return getToken(planParser.FloatingConstant, 0); }
		public FloatingContext(ConstExprContext ctx) { copyFrom(ctx); }
	}
	public static class ConstAddSubContext extends ConstExprContext {
		public Token op;
		public List<ConstExprContext> constExpr() {
			return getRuleContexts(ConstExprContext.class);
		}
		public ConstExprContext constExpr(int i) {
			return getRuleContext(ConstExprContext.class,i);
		}
		public TerminalNode ADD() { return getToken(planParser.ADD, 0); }
		public TerminalNode SUB() { return getToken(planParser.SUB, 0); }
		public ConstAddSubContext(ConstExprContext ctx) { copyFrom(ctx); }
	}
	public static class ConstMulDivModContext extends ConstExprContext {
		public Token op;
		public List<ConstExprContext> constExpr() {
			return getRuleContexts(ConstExprContext.class);
		}
		public ConstExprContext constExpr(int i) {
			return getRuleContext(ConstExprContext.class,i);
		}
		public TerminalNode MUL() { return getToken(planParser.MUL, 0); }
		public TerminalNode DIV() { return getToken(planParser.DIV, 0); }
		public TerminalNode MOD() { return getToken(planParser.MOD, 0); }
		public ConstMulDivModContext(ConstExprContext ctx) { copyFrom(ctx); }
	}
	public static class ConstParenthesisContext extends ConstExprContext {
		public ConstExprContext constExpr() {
			return getRuleContext(ConstExprContext.class,0);
		}
		public ConstParenthesisContext(ConstExprContext ctx) { copyFrom(ctx); }
	}
	public static class IntegerContext extends ConstExprContext {
		public TerminalNode IntegerConstant() { return getToken(planParser.IntegerConstant, 0); }
		public IntegerContext(ConstExprContext ctx) { copyFrom(ctx); }
	}
	public static class ConstLogiAndContext extends ConstExprContext {
		public List<ConstExprContext> constExpr() {
			return getRuleContexts(ConstExprContext.class);
		}
		public ConstExprContext constExpr(int i) {
			return getRuleContext(ConstExprContext.class,i);
		}
		public TerminalNode AND() { return getToken(planParser.AND, 0); }
		public ConstLogiAndContext(ConstExprContext ctx) { copyFrom(ctx); }
	}
	public static class ConstBitOrContext extends ConstExprContext {
		public List<ConstExprContext> constExpr() {
			return getRuleContexts(ConstExprContext.class);
		}
		public ConstExprContext constExpr(int i) {
			return getRuleContext(ConstExprContext.class,i);
		}
		public TerminalNode BOR() { return getToken(planParser.BOR, 0); }
		public ConstBitOrContext(ConstExprContext ctx) { copyFrom(ctx); }
	}
	public static class ConstShiftContext extends ConstExprContext {
		public Token op;
		public List<ConstExprContext> constExpr() {
			return getRuleContexts(ConstExprContext.class);
		}
		public ConstExprContext constExpr(int i) {
			return getRuleContext(ConstExprContext.class,i);
		}
		public TerminalNode SHL() { return getToken(planParser.SHL, 0); }
		public TerminalNode SHR() { return getToken(planParser.SHR, 0); }
		public ConstShiftContext(ConstExprContext ctx) { copyFrom(ctx); }
	}
	public static class ConstLogiOrContext extends ConstExprContext {
		public List<ConstExprContext> constExpr() {
			return getRuleContexts(ConstExprContext.class);
		}
		public ConstExprContext constExpr(int i) {
			return getRuleContext(ConstExprContext.class,i);
		}
		public TerminalNode OR() { return getToken(planParser.OR, 0); }
		public ConstLogiOrContext(ConstExprContext ctx) { copyFrom(ctx); }
	}
	public static class BooleanContext extends ConstExprContext {
		public TerminalNode BooleanConstant() { return getToken(planParser.BooleanConstant, 0); }
		public BooleanContext(ConstExprContext ctx) { copyFrom(ctx); }
	}
	public static class ConstBitAndContext extends ConstExprContext {
		public List<ConstExprContext> constExpr() {
			return getRuleContexts(ConstExprContext.class);
		}
		public ConstExprContext constExpr(int i) {
			return getRuleContext(ConstExprContext.class,i);
		}
		public TerminalNode BAND() { return getToken(planParser.BAND, 0); }
		public ConstBitAndContext(ConstExprContext ctx) { copyFrom(ctx); }
	}

	public final ConstExprContext constExpr() throws RecognitionException {
		return constExpr(0);
	}

	private ConstExprContext constExpr(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		ConstExprContext _localctx = new ConstExprContext(_ctx, _parentState);
		ConstExprContext _prevctx = _localctx;
		int _startState = 0;
		enterRecursionRule(_localctx, 0, RULE_constExpr, _p);
		int _la;
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(19);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case IntegerConstant:
				{
				_localctx = new IntegerContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;

				setState(9);
				match(IntegerConstant);
				}
				break;
			case FloatingConstant:
				{
				_localctx = new FloatingContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(10);
				match(FloatingConstant);
				}
				break;
			case BooleanConstant:
				{
				_localctx = new BooleanContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(11);
				match(BooleanConstant);
				}
				break;
			case StringLiteral:
				{
				_localctx = new StringContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(12);
				match(StringLiteral);
				}
				break;
			case T__0:
				{
				_localctx = new ConstParenthesisContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(13);
				match(T__0);
				setState(14);
				constExpr(0);
				setState(15);
				match(T__1);
				}
				break;
			case ADD:
			case SUB:
			case BNOT:
			case NOT:
				{
				_localctx = new ConstUnaryContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(17);
				((ConstUnaryContext)_localctx).op = _input.LT(1);
				_la = _input.LA(1);
				if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << ADD) | (1L << SUB) | (1L << BNOT) | (1L << NOT))) != 0)) ) {
					((ConstUnaryContext)_localctx).op = (Token)_errHandler.recoverInline(this);
				}
				else {
					if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
					_errHandler.reportMatch(this);
					consume();
				}
				setState(18);
				constExpr(11);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			_ctx.stop = _input.LT(-1);
			setState(53);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,2,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					setState(51);
					_errHandler.sync(this);
					switch ( getInterpreter().adaptivePredict(_input,1,_ctx) ) {
					case 1:
						{
						_localctx = new ConstMulDivModContext(new ConstExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_constExpr);
						setState(21);
						if (!(precpred(_ctx, 10))) throw new FailedPredicateException(this, "precpred(_ctx, 10)");
						setState(22);
						((ConstMulDivModContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << MUL) | (1L << DIV) | (1L << MOD))) != 0)) ) {
							((ConstMulDivModContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(23);
						constExpr(11);
						}
						break;
					case 2:
						{
						_localctx = new ConstAddSubContext(new ConstExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_constExpr);
						setState(24);
						if (!(precpred(_ctx, 9))) throw new FailedPredicateException(this, "precpred(_ctx, 9)");
						setState(25);
						((ConstAddSubContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !(_la==ADD || _la==SUB) ) {
							((ConstAddSubContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(26);
						constExpr(10);
						}
						break;
					case 3:
						{
						_localctx = new ConstShiftContext(new ConstExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_constExpr);
						setState(27);
						if (!(precpred(_ctx, 8))) throw new FailedPredicateException(this, "precpred(_ctx, 8)");
						setState(28);
						((ConstShiftContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !(_la==SHL || _la==SHR) ) {
							((ConstShiftContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(29);
						constExpr(9);
						}
						break;
					case 4:
						{
						_localctx = new ConstRelationalContext(new ConstExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_constExpr);
						setState(30);
						if (!(precpred(_ctx, 7))) throw new FailedPredicateException(this, "precpred(_ctx, 7)");
						setState(31);
						((ConstRelationalContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << LT) | (1L << LE) | (1L << GT) | (1L << GE))) != 0)) ) {
							((ConstRelationalContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(32);
						constExpr(8);
						}
						break;
					case 5:
						{
						_localctx = new ConstEqualityContext(new ConstExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_constExpr);
						setState(33);
						if (!(precpred(_ctx, 6))) throw new FailedPredicateException(this, "precpred(_ctx, 6)");
						setState(34);
						((ConstEqualityContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !(_la==EQ || _la==NE) ) {
							((ConstEqualityContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(35);
						constExpr(7);
						}
						break;
					case 6:
						{
						_localctx = new ConstBitAndContext(new ConstExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_constExpr);
						setState(36);
						if (!(precpred(_ctx, 5))) throw new FailedPredicateException(this, "precpred(_ctx, 5)");
						setState(37);
						match(BAND);
						setState(38);
						constExpr(6);
						}
						break;
					case 7:
						{
						_localctx = new ConstBitXorContext(new ConstExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_constExpr);
						setState(39);
						if (!(precpred(_ctx, 4))) throw new FailedPredicateException(this, "precpred(_ctx, 4)");
						setState(40);
						match(BXOR);
						setState(41);
						constExpr(5);
						}
						break;
					case 8:
						{
						_localctx = new ConstBitOrContext(new ConstExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_constExpr);
						setState(42);
						if (!(precpred(_ctx, 3))) throw new FailedPredicateException(this, "precpred(_ctx, 3)");
						setState(43);
						match(BOR);
						setState(44);
						constExpr(4);
						}
						break;
					case 9:
						{
						_localctx = new ConstLogiAndContext(new ConstExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_constExpr);
						setState(45);
						if (!(precpred(_ctx, 2))) throw new FailedPredicateException(this, "precpred(_ctx, 2)");
						setState(46);
						match(AND);
						setState(47);
						constExpr(3);
						}
						break;
					case 10:
						{
						_localctx = new ConstLogiOrContext(new ConstExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_constExpr);
						setState(48);
						if (!(precpred(_ctx, 1))) throw new FailedPredicateException(this, "precpred(_ctx, 1)");
						setState(49);
						match(OR);
						setState(50);
						constExpr(2);
						}
						break;
					}
					} 
				}
				setState(55);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,2,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	public static class ExprContext extends ParserRuleContext {
		public ExprContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_expr; }
	 
		public ExprContext() { }
		public void copyFrom(ExprContext ctx) {
			super.copyFrom(ctx);
		}
	}
	public static class CastContext extends ExprContext {
		public TypeNameContext typeName() {
			return getRuleContext(TypeNameContext.class,0);
		}
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public CastContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class ShiftContext extends ExprContext {
		public Token op;
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode SHL() { return getToken(planParser.SHL, 0); }
		public TerminalNode SHR() { return getToken(planParser.SHR, 0); }
		public ShiftContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class BitOrContext extends ExprContext {
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode BOR() { return getToken(planParser.BOR, 0); }
		public BitOrContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class ConstExpressionContext extends ExprContext {
		public ConstExprContext constExpr() {
			return getRuleContext(ConstExprContext.class,0);
		}
		public ConstExpressionContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class AddSubContext extends ExprContext {
		public Token op;
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode ADD() { return getToken(planParser.ADD, 0); }
		public TerminalNode SUB() { return getToken(planParser.SUB, 0); }
		public AddSubContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class RelationalContext extends ExprContext {
		public Token op;
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode LT() { return getToken(planParser.LT, 0); }
		public TerminalNode LE() { return getToken(planParser.LE, 0); }
		public TerminalNode GT() { return getToken(planParser.GT, 0); }
		public TerminalNode GE() { return getToken(planParser.GE, 0); }
		public RelationalContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class TermContext extends ExprContext {
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public TerminalNode IN() { return getToken(planParser.IN, 0); }
		public TermListContext termList() {
			return getRuleContext(TermListContext.class,0);
		}
		public TermContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class RangeContext extends ExprContext {
		public Token op1;
		public Token op2;
		public List<ConstExprContext> constExpr() {
			return getRuleContexts(ConstExprContext.class);
		}
		public ConstExprContext constExpr(int i) {
			return getRuleContext(ConstExprContext.class,i);
		}
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public List<TerminalNode> LT() { return getTokens(planParser.LT); }
		public TerminalNode LT(int i) {
			return getToken(planParser.LT, i);
		}
		public List<TerminalNode> LE() { return getTokens(planParser.LE); }
		public TerminalNode LE(int i) {
			return getToken(planParser.LE, i);
		}
		public RangeContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class LogiAndContext extends ExprContext {
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode AND() { return getToken(planParser.AND, 0); }
		public LogiAndContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class UnaryContext extends ExprContext {
		public Token op;
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public TerminalNode ADD() { return getToken(planParser.ADD, 0); }
		public TerminalNode SUB() { return getToken(planParser.SUB, 0); }
		public TerminalNode BNOT() { return getToken(planParser.BNOT, 0); }
		public TerminalNode NOT() { return getToken(planParser.NOT, 0); }
		public UnaryContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class ParenthesisContext extends ExprContext {
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public ParenthesisContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class MulDivModContext extends ExprContext {
		public Token op;
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode MUL() { return getToken(planParser.MUL, 0); }
		public TerminalNode DIV() { return getToken(planParser.DIV, 0); }
		public TerminalNode MOD() { return getToken(planParser.MOD, 0); }
		public MulDivModContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class IdentifierContext extends ExprContext {
		public TerminalNode Identifier() { return getToken(planParser.Identifier, 0); }
		public IdentifierContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class BitXorContext extends ExprContext {
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode BXOR() { return getToken(planParser.BXOR, 0); }
		public BitXorContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class LogiOrContext extends ExprContext {
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode OR() { return getToken(planParser.OR, 0); }
		public LogiOrContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class BitAndContext extends ExprContext {
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode BAND() { return getToken(planParser.BAND, 0); }
		public BitAndContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class EqualityContext extends ExprContext {
		public Token op;
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode EQ() { return getToken(planParser.EQ, 0); }
		public TerminalNode NE() { return getToken(planParser.NE, 0); }
		public EqualityContext(ExprContext ctx) { copyFrom(ctx); }
	}

	public final ExprContext expr() throws RecognitionException {
		return expr(0);
	}

	private ExprContext expr(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		ExprContext _localctx = new ExprContext(_ctx, _parentState);
		ExprContext _prevctx = _localctx;
		int _startState = 2;
		enterRecursionRule(_localctx, 2, RULE_expr, _p);
		int _la;
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(76);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,3,_ctx) ) {
			case 1:
				{
				_localctx = new IdentifierContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;

				setState(57);
				match(Identifier);
				}
				break;
			case 2:
				{
				_localctx = new ConstExpressionContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(58);
				constExpr(0);
				}
				break;
			case 3:
				{
				_localctx = new ParenthesisContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(59);
				match(T__0);
				setState(60);
				expr(0);
				setState(61);
				match(T__1);
				}
				break;
			case 4:
				{
				_localctx = new UnaryContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(63);
				((UnaryContext)_localctx).op = _input.LT(1);
				_la = _input.LA(1);
				if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << ADD) | (1L << SUB) | (1L << BNOT) | (1L << NOT))) != 0)) ) {
					((UnaryContext)_localctx).op = (Token)_errHandler.recoverInline(this);
				}
				else {
					if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
					_errHandler.reportMatch(this);
					consume();
				}
				setState(64);
				expr(14);
				}
				break;
			case 5:
				{
				_localctx = new CastContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(65);
				match(T__0);
				setState(66);
				typeName();
				setState(67);
				match(T__1);
				setState(68);
				expr(13);
				}
				break;
			case 6:
				{
				_localctx = new RangeContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(70);
				constExpr(0);
				setState(71);
				((RangeContext)_localctx).op1 = _input.LT(1);
				_la = _input.LA(1);
				if ( !(_la==LT || _la==LE) ) {
					((RangeContext)_localctx).op1 = (Token)_errHandler.recoverInline(this);
				}
				else {
					if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
					_errHandler.reportMatch(this);
					consume();
				}
				setState(72);
				expr(0);
				setState(73);
				((RangeContext)_localctx).op2 = _input.LT(1);
				_la = _input.LA(1);
				if ( !(_la==LT || _la==LE) ) {
					((RangeContext)_localctx).op2 = (Token)_errHandler.recoverInline(this);
				}
				else {
					if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
					_errHandler.reportMatch(this);
					consume();
				}
				setState(74);
				constExpr(0);
				}
				break;
			}
			_ctx.stop = _input.LT(-1);
			setState(113);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,5,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					setState(111);
					_errHandler.sync(this);
					switch ( getInterpreter().adaptivePredict(_input,4,_ctx) ) {
					case 1:
						{
						_localctx = new MulDivModContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(78);
						if (!(precpred(_ctx, 12))) throw new FailedPredicateException(this, "precpred(_ctx, 12)");
						setState(79);
						((MulDivModContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << MUL) | (1L << DIV) | (1L << MOD))) != 0)) ) {
							((MulDivModContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(80);
						expr(13);
						}
						break;
					case 2:
						{
						_localctx = new AddSubContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(81);
						if (!(precpred(_ctx, 11))) throw new FailedPredicateException(this, "precpred(_ctx, 11)");
						setState(82);
						((AddSubContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !(_la==ADD || _la==SUB) ) {
							((AddSubContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(83);
						expr(12);
						}
						break;
					case 3:
						{
						_localctx = new ShiftContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(84);
						if (!(precpred(_ctx, 10))) throw new FailedPredicateException(this, "precpred(_ctx, 10)");
						setState(85);
						((ShiftContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !(_la==SHL || _la==SHR) ) {
							((ShiftContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(86);
						expr(11);
						}
						break;
					case 4:
						{
						_localctx = new RelationalContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(87);
						if (!(precpred(_ctx, 7))) throw new FailedPredicateException(this, "precpred(_ctx, 7)");
						setState(88);
						((RelationalContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << LT) | (1L << LE) | (1L << GT) | (1L << GE))) != 0)) ) {
							((RelationalContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(89);
						expr(8);
						}
						break;
					case 5:
						{
						_localctx = new EqualityContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(90);
						if (!(precpred(_ctx, 6))) throw new FailedPredicateException(this, "precpred(_ctx, 6)");
						setState(91);
						((EqualityContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !(_la==EQ || _la==NE) ) {
							((EqualityContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(92);
						expr(7);
						}
						break;
					case 6:
						{
						_localctx = new BitAndContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(93);
						if (!(precpred(_ctx, 5))) throw new FailedPredicateException(this, "precpred(_ctx, 5)");
						setState(94);
						match(BAND);
						setState(95);
						expr(6);
						}
						break;
					case 7:
						{
						_localctx = new BitXorContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(96);
						if (!(precpred(_ctx, 4))) throw new FailedPredicateException(this, "precpred(_ctx, 4)");
						setState(97);
						match(BXOR);
						setState(98);
						expr(5);
						}
						break;
					case 8:
						{
						_localctx = new BitOrContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(99);
						if (!(precpred(_ctx, 3))) throw new FailedPredicateException(this, "precpred(_ctx, 3)");
						setState(100);
						match(BOR);
						setState(101);
						expr(4);
						}
						break;
					case 9:
						{
						_localctx = new LogiAndContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(102);
						if (!(precpred(_ctx, 2))) throw new FailedPredicateException(this, "precpred(_ctx, 2)");
						setState(103);
						match(AND);
						setState(104);
						expr(3);
						}
						break;
					case 10:
						{
						_localctx = new LogiOrContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(105);
						if (!(precpred(_ctx, 1))) throw new FailedPredicateException(this, "precpred(_ctx, 1)");
						setState(106);
						match(OR);
						setState(107);
						expr(2);
						}
						break;
					case 11:
						{
						_localctx = new TermContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(108);
						if (!(precpred(_ctx, 8))) throw new FailedPredicateException(this, "precpred(_ctx, 8)");
						setState(109);
						match(IN);
						setState(110);
						termList();
						}
						break;
					}
					} 
				}
				setState(115);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,5,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	public static class TermListContext extends ParserRuleContext {
		public List<ConstExprContext> constExpr() {
			return getRuleContexts(ConstExprContext.class);
		}
		public ConstExprContext constExpr(int i) {
			return getRuleContext(ConstExprContext.class,i);
		}
		public TermListContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_termList; }
	}

	public final TermListContext termList() throws RecognitionException {
		TermListContext _localctx = new TermListContext(_ctx, getState());
		enterRule(_localctx, 4, RULE_termList);
		int _la;
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(116);
			match(T__2);
			setState(117);
			constExpr(0);
			setState(122);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,6,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					{
					{
					setState(118);
					match(T__3);
					setState(119);
					constExpr(0);
					}
					} 
				}
				setState(124);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,6,_ctx);
			}
			setState(126);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__3) {
				{
				setState(125);
				match(T__3);
				}
			}

			setState(128);
			match(T__4);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class TypeNameContext extends ParserRuleContext {
		public TerminalNode BOOL() { return getToken(planParser.BOOL, 0); }
		public TerminalNode INT8() { return getToken(planParser.INT8, 0); }
		public TerminalNode INT16() { return getToken(planParser.INT16, 0); }
		public TerminalNode INT32() { return getToken(planParser.INT32, 0); }
		public TerminalNode INT64() { return getToken(planParser.INT64, 0); }
		public TerminalNode FLOAT() { return getToken(planParser.FLOAT, 0); }
		public TerminalNode DOUBLE() { return getToken(planParser.DOUBLE, 0); }
		public TypeNameContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_typeName; }
	}

	public final TypeNameContext typeName() throws RecognitionException {
		TypeNameContext _localctx = new TypeNameContext(_ctx, getState());
		enterRule(_localctx, 6, RULE_typeName);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(130);
			_la = _input.LA(1);
			if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << BOOL) | (1L << INT8) | (1L << INT16) | (1L << INT32) | (1L << INT64) | (1L << FLOAT) | (1L << DOUBLE))) != 0)) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public boolean sempred(RuleContext _localctx, int ruleIndex, int predIndex) {
		switch (ruleIndex) {
		case 0:
			return constExpr_sempred((ConstExprContext)_localctx, predIndex);
		case 1:
			return expr_sempred((ExprContext)_localctx, predIndex);
		}
		return true;
	}
	private boolean constExpr_sempred(ConstExprContext _localctx, int predIndex) {
		switch (predIndex) {
		case 0:
			return precpred(_ctx, 10);
		case 1:
			return precpred(_ctx, 9);
		case 2:
			return precpred(_ctx, 8);
		case 3:
			return precpred(_ctx, 7);
		case 4:
			return precpred(_ctx, 6);
		case 5:
			return precpred(_ctx, 5);
		case 6:
			return precpred(_ctx, 4);
		case 7:
			return precpred(_ctx, 3);
		case 8:
			return precpred(_ctx, 2);
		case 9:
			return precpred(_ctx, 1);
		}
		return true;
	}
	private boolean expr_sempred(ExprContext _localctx, int predIndex) {
		switch (predIndex) {
		case 10:
			return precpred(_ctx, 12);
		case 11:
			return precpred(_ctx, 11);
		case 12:
			return precpred(_ctx, 10);
		case 13:
			return precpred(_ctx, 7);
		case 14:
			return precpred(_ctx, 6);
		case 15:
			return precpred(_ctx, 5);
		case 16:
			return precpred(_ctx, 4);
		case 17:
			return precpred(_ctx, 3);
		case 18:
			return precpred(_ctx, 2);
		case 19:
			return precpred(_ctx, 1);
		case 20:
			return precpred(_ctx, 8);
		}
		return true;
	}

	public static final String _serializedATN =
		"\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\3,\u0087\4\2\t\2\4"+
		"\3\t\3\4\4\t\4\4\5\t\5\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\5\2"+
		"\26\n\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3"+
		"\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\7\2\66\n\2"+
		"\f\2\16\29\13\2\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3"+
		"\3\3\3\3\3\3\3\3\3\3\3\3\3\5\3O\n\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3"+
		"\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3"+
		"\3\3\3\3\3\3\3\3\3\3\3\3\3\3\7\3r\n\3\f\3\16\3u\13\3\3\4\3\4\3\4\3\4\7"+
		"\4{\n\4\f\4\16\4~\13\4\3\4\5\4\u0081\n\4\3\4\3\4\3\5\3\5\3\5\2\4\2\4\6"+
		"\2\4\6\b\2\n\4\2\25\26!\"\3\2\27\31\3\2\25\26\3\2\32\33\3\2\17\22\3\2"+
		"\23\24\3\2\17\20\3\2\b\16\2\u00a3\2\25\3\2\2\2\4N\3\2\2\2\6v\3\2\2\2\b"+
		"\u0084\3\2\2\2\n\13\b\2\1\2\13\26\7(\2\2\f\26\7)\2\2\r\26\7*\2\2\16\26"+
		"\7\'\2\2\17\20\7\3\2\2\20\21\5\2\2\2\21\22\7\4\2\2\22\26\3\2\2\2\23\24"+
		"\t\2\2\2\24\26\5\2\2\r\25\n\3\2\2\2\25\f\3\2\2\2\25\r\3\2\2\2\25\16\3"+
		"\2\2\2\25\17\3\2\2\2\25\23\3\2\2\2\26\67\3\2\2\2\27\30\f\f\2\2\30\31\t"+
		"\3\2\2\31\66\5\2\2\r\32\33\f\13\2\2\33\34\t\4\2\2\34\66\5\2\2\f\35\36"+
		"\f\n\2\2\36\37\t\5\2\2\37\66\5\2\2\13 !\f\t\2\2!\"\t\6\2\2\"\66\5\2\2"+
		"\n#$\f\b\2\2$%\t\7\2\2%\66\5\2\2\t&\'\f\7\2\2\'(\7\34\2\2(\66\5\2\2\b"+
		")*\f\6\2\2*+\7\36\2\2+\66\5\2\2\7,-\f\5\2\2-.\7\35\2\2.\66\5\2\2\6/\60"+
		"\f\4\2\2\60\61\7\37\2\2\61\66\5\2\2\5\62\63\f\3\2\2\63\64\7 \2\2\64\66"+
		"\5\2\2\4\65\27\3\2\2\2\65\32\3\2\2\2\65\35\3\2\2\2\65 \3\2\2\2\65#\3\2"+
		"\2\2\65&\3\2\2\2\65)\3\2\2\2\65,\3\2\2\2\65/\3\2\2\2\65\62\3\2\2\2\66"+
		"9\3\2\2\2\67\65\3\2\2\2\678\3\2\2\28\3\3\2\2\29\67\3\2\2\2:;\b\3\1\2;"+
		"O\7&\2\2<O\5\2\2\2=>\7\3\2\2>?\5\4\3\2?@\7\4\2\2@O\3\2\2\2AB\t\2\2\2B"+
		"O\5\4\3\20CD\7\3\2\2DE\5\b\5\2EF\7\4\2\2FG\5\4\3\17GO\3\2\2\2HI\5\2\2"+
		"\2IJ\t\b\2\2JK\5\4\3\2KL\t\b\2\2LM\5\2\2\2MO\3\2\2\2N:\3\2\2\2N<\3\2\2"+
		"\2N=\3\2\2\2NA\3\2\2\2NC\3\2\2\2NH\3\2\2\2Os\3\2\2\2PQ\f\16\2\2QR\t\3"+
		"\2\2Rr\5\4\3\17ST\f\r\2\2TU\t\4\2\2Ur\5\4\3\16VW\f\f\2\2WX\t\5\2\2Xr\5"+
		"\4\3\rYZ\f\t\2\2Z[\t\6\2\2[r\5\4\3\n\\]\f\b\2\2]^\t\7\2\2^r\5\4\3\t_`"+
		"\f\7\2\2`a\7\34\2\2ar\5\4\3\bbc\f\6\2\2cd\7\36\2\2dr\5\4\3\7ef\f\5\2\2"+
		"fg\7\35\2\2gr\5\4\3\6hi\f\4\2\2ij\7\37\2\2jr\5\4\3\5kl\f\3\2\2lm\7 \2"+
		"\2mr\5\4\3\4no\f\n\2\2op\7#\2\2pr\5\6\4\2qP\3\2\2\2qS\3\2\2\2qV\3\2\2"+
		"\2qY\3\2\2\2q\\\3\2\2\2q_\3\2\2\2qb\3\2\2\2qe\3\2\2\2qh\3\2\2\2qk\3\2"+
		"\2\2qn\3\2\2\2ru\3\2\2\2sq\3\2\2\2st\3\2\2\2t\5\3\2\2\2us\3\2\2\2vw\7"+
		"\5\2\2w|\5\2\2\2xy\7\6\2\2y{\5\2\2\2zx\3\2\2\2{~\3\2\2\2|z\3\2\2\2|}\3"+
		"\2\2\2}\u0080\3\2\2\2~|\3\2\2\2\177\u0081\7\6\2\2\u0080\177\3\2\2\2\u0080"+
		"\u0081\3\2\2\2\u0081\u0082\3\2\2\2\u0082\u0083\7\7\2\2\u0083\7\3\2\2\2"+
		"\u0084\u0085\t\t\2\2\u0085\t\3\2\2\2\n\25\65\67Nqs|\u0080";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}