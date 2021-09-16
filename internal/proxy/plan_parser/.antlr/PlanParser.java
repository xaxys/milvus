// Generated from /home/xa/milvus/internal/proxy/plan_parser/Plan.g4 by ANTLR 4.8
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.misc.*;
import org.antlr.v4.runtime.tree.*;
import java.util.List;
import java.util.Iterator;
import java.util.ArrayList;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast"})
public class PlanParser extends Parser {
	static { RuntimeMetaData.checkVersion("4.8", RuntimeMetaData.VERSION); }

	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		T__0=1, T__1=2, T__2=3, T__3=4, T__4=5, BOOL=6, INT8=7, INT16=8, INT32=9, 
		INT64=10, FLOAT=11, DOUBLE=12, LT=13, LE=14, GT=15, GE=16, EQ=17, NE=18, 
		ADD=19, SUB=20, MUL=21, DIV=22, MOD=23, POW=24, SHL=25, SHR=26, BAND=27, 
		BOR=28, BXOR=29, AND=30, OR=31, BNOT=32, NOT=33, IN=34, NIN=35, BooleanConstant=36, 
		IntegerConstant=37, FloatingConstant=38, Identifier=39, StringLiteral=40, 
		Whitespace=41, Newline=42;
	public static final int
		RULE_expr = 0, RULE_typeName = 1;
	private static String[] makeRuleNames() {
		return new String[] {
			"expr", "typeName"
		};
	}
	public static final String[] ruleNames = makeRuleNames();

	private static String[] makeLiteralNames() {
		return new String[] {
			null, "'('", "')'", "'['", "','", "']'", "'bool'", "'int8'", "'int16'", 
			"'int32'", "'int64'", "'float'", "'double'", "'<'", "'<='", "'>'", "'>='", 
			"'=='", "'!='", "'+'", "'-'", "'*'", "'/'", "'%'", "'**'", "'<<'", "'>>'", 
			"'&'", "'|'", "'^'", null, null, "'~'", null, "'in'", "'not in'"
		};
	}
	private static final String[] _LITERAL_NAMES = makeLiteralNames();
	private static String[] makeSymbolicNames() {
		return new String[] {
			null, null, null, null, null, null, "BOOL", "INT8", "INT16", "INT32", 
			"INT64", "FLOAT", "DOUBLE", "LT", "LE", "GT", "GE", "EQ", "NE", "ADD", 
			"SUB", "MUL", "DIV", "MOD", "POW", "SHL", "SHR", "BAND", "BOR", "BXOR", 
			"AND", "OR", "BNOT", "NOT", "IN", "NIN", "BooleanConstant", "IntegerConstant", 
			"FloatingConstant", "Identifier", "StringLiteral", "Whitespace", "Newline"
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
	public String getGrammarFileName() { return "Plan.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public ATN getATN() { return _ATN; }

	public PlanParser(TokenStream input) {
		super(input);
		_interp = new ParserATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
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
		public TerminalNode SHL() { return getToken(PlanParser.SHL, 0); }
		public TerminalNode SHR() { return getToken(PlanParser.SHR, 0); }
		public ShiftContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class BitOrContext extends ExprContext {
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode BOR() { return getToken(PlanParser.BOR, 0); }
		public BitOrContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class AddSubContext extends ExprContext {
		public Token op;
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode ADD() { return getToken(PlanParser.ADD, 0); }
		public TerminalNode SUB() { return getToken(PlanParser.SUB, 0); }
		public AddSubContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class ParensContext extends ExprContext {
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public ParensContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class RelationalContext extends ExprContext {
		public Token op;
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode LT() { return getToken(PlanParser.LT, 0); }
		public TerminalNode LE() { return getToken(PlanParser.LE, 0); }
		public TerminalNode GT() { return getToken(PlanParser.GT, 0); }
		public TerminalNode GE() { return getToken(PlanParser.GE, 0); }
		public RelationalContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class StringContext extends ExprContext {
		public TerminalNode StringLiteral() { return getToken(PlanParser.StringLiteral, 0); }
		public StringContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class TermContext extends ExprContext {
		public Token op;
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode IN() { return getToken(PlanParser.IN, 0); }
		public TerminalNode NIN() { return getToken(PlanParser.NIN, 0); }
		public TermContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class FloatingContext extends ExprContext {
		public TerminalNode FloatingConstant() { return getToken(PlanParser.FloatingConstant, 0); }
		public FloatingContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class RangeContext extends ExprContext {
		public Token op1;
		public Token op2;
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public List<TerminalNode> LT() { return getTokens(PlanParser.LT); }
		public TerminalNode LT(int i) {
			return getToken(PlanParser.LT, i);
		}
		public List<TerminalNode> LE() { return getTokens(PlanParser.LE); }
		public TerminalNode LE(int i) {
			return getToken(PlanParser.LE, i);
		}
		public RangeContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class UnaryContext extends ExprContext {
		public Token op;
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public TerminalNode ADD() { return getToken(PlanParser.ADD, 0); }
		public TerminalNode SUB() { return getToken(PlanParser.SUB, 0); }
		public TerminalNode BNOT() { return getToken(PlanParser.BNOT, 0); }
		public TerminalNode NOT() { return getToken(PlanParser.NOT, 0); }
		public UnaryContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class LogicalOrContext extends ExprContext {
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode OR() { return getToken(PlanParser.OR, 0); }
		public LogicalOrContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class IntegerContext extends ExprContext {
		public TerminalNode IntegerConstant() { return getToken(PlanParser.IntegerConstant, 0); }
		public IntegerContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class MulDivModContext extends ExprContext {
		public Token op;
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode MUL() { return getToken(PlanParser.MUL, 0); }
		public TerminalNode DIV() { return getToken(PlanParser.DIV, 0); }
		public TerminalNode MOD() { return getToken(PlanParser.MOD, 0); }
		public MulDivModContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class IdentifierContext extends ExprContext {
		public TerminalNode Identifier() { return getToken(PlanParser.Identifier, 0); }
		public IdentifierContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class BitXorContext extends ExprContext {
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode BXOR() { return getToken(PlanParser.BXOR, 0); }
		public BitXorContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class BitAndContext extends ExprContext {
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode BAND() { return getToken(PlanParser.BAND, 0); }
		public BitAndContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class LogicalAndContext extends ExprContext {
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode AND() { return getToken(PlanParser.AND, 0); }
		public LogicalAndContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class EqualityContext extends ExprContext {
		public Token op;
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode EQ() { return getToken(PlanParser.EQ, 0); }
		public TerminalNode NE() { return getToken(PlanParser.NE, 0); }
		public EqualityContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class BooleanContext extends ExprContext {
		public TerminalNode BooleanConstant() { return getToken(PlanParser.BooleanConstant, 0); }
		public BooleanContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class PowerContext extends ExprContext {
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode POW() { return getToken(PlanParser.POW, 0); }
		public PowerContext(ExprContext ctx) { copyFrom(ctx); }
	}

	public final ExprContext expr() throws RecognitionException {
		return expr(0);
	}

	private ExprContext expr(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		ExprContext _localctx = new ExprContext(_ctx, _parentState);
		ExprContext _prevctx = _localctx;
		int _startState = 0;
		enterRecursionRule(_localctx, 0, RULE_expr, _p);
		int _la;
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(21);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,0,_ctx) ) {
			case 1:
				{
				_localctx = new IntegerContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;

				setState(5);
				match(IntegerConstant);
				}
				break;
			case 2:
				{
				_localctx = new FloatingContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(6);
				match(FloatingConstant);
				}
				break;
			case 3:
				{
				_localctx = new BooleanContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(7);
				match(BooleanConstant);
				}
				break;
			case 4:
				{
				_localctx = new StringContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(8);
				match(StringLiteral);
				}
				break;
			case 5:
				{
				_localctx = new IdentifierContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(9);
				match(Identifier);
				}
				break;
			case 6:
				{
				_localctx = new ParensContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(10);
				match(T__0);
				setState(11);
				expr(0);
				setState(12);
				match(T__1);
				}
				break;
			case 7:
				{
				_localctx = new UnaryContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(14);
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
				setState(15);
				expr(14);
				}
				break;
			case 8:
				{
				_localctx = new CastContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(16);
				match(T__0);
				setState(17);
				typeName();
				setState(18);
				match(T__1);
				setState(19);
				expr(13);
				}
				break;
			}
			_ctx.stop = _input.LT(-1);
			setState(80);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,4,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					setState(78);
					_errHandler.sync(this);
					switch ( getInterpreter().adaptivePredict(_input,3,_ctx) ) {
					case 1:
						{
						_localctx = new PowerContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(23);
						if (!(precpred(_ctx, 15))) throw new FailedPredicateException(this, "precpred(_ctx, 15)");
						setState(24);
						match(POW);
						setState(25);
						expr(16);
						}
						break;
					case 2:
						{
						_localctx = new MulDivModContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(26);
						if (!(precpred(_ctx, 12))) throw new FailedPredicateException(this, "precpred(_ctx, 12)");
						setState(27);
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
						setState(28);
						expr(13);
						}
						break;
					case 3:
						{
						_localctx = new AddSubContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(29);
						if (!(precpred(_ctx, 11))) throw new FailedPredicateException(this, "precpred(_ctx, 11)");
						setState(30);
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
						setState(31);
						expr(12);
						}
						break;
					case 4:
						{
						_localctx = new ShiftContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(32);
						if (!(precpred(_ctx, 10))) throw new FailedPredicateException(this, "precpred(_ctx, 10)");
						setState(33);
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
						setState(34);
						expr(11);
						}
						break;
					case 5:
						{
						_localctx = new RangeContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(35);
						if (!(precpred(_ctx, 8))) throw new FailedPredicateException(this, "precpred(_ctx, 8)");
						setState(36);
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
						setState(37);
						expr(0);
						setState(38);
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
						setState(39);
						expr(9);
						}
						break;
					case 6:
						{
						_localctx = new RelationalContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(41);
						if (!(precpred(_ctx, 7))) throw new FailedPredicateException(this, "precpred(_ctx, 7)");
						setState(42);
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
						setState(43);
						expr(8);
						}
						break;
					case 7:
						{
						_localctx = new EqualityContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(44);
						if (!(precpred(_ctx, 6))) throw new FailedPredicateException(this, "precpred(_ctx, 6)");
						setState(45);
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
						setState(46);
						expr(7);
						}
						break;
					case 8:
						{
						_localctx = new BitAndContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(47);
						if (!(precpred(_ctx, 5))) throw new FailedPredicateException(this, "precpred(_ctx, 5)");
						setState(48);
						match(BAND);
						setState(49);
						expr(6);
						}
						break;
					case 9:
						{
						_localctx = new BitXorContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(50);
						if (!(precpred(_ctx, 4))) throw new FailedPredicateException(this, "precpred(_ctx, 4)");
						setState(51);
						match(BXOR);
						setState(52);
						expr(5);
						}
						break;
					case 10:
						{
						_localctx = new BitOrContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(53);
						if (!(precpred(_ctx, 3))) throw new FailedPredicateException(this, "precpred(_ctx, 3)");
						setState(54);
						match(BOR);
						setState(55);
						expr(4);
						}
						break;
					case 11:
						{
						_localctx = new LogicalAndContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(56);
						if (!(precpred(_ctx, 2))) throw new FailedPredicateException(this, "precpred(_ctx, 2)");
						setState(57);
						match(AND);
						setState(58);
						expr(3);
						}
						break;
					case 12:
						{
						_localctx = new LogicalOrContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(59);
						if (!(precpred(_ctx, 1))) throw new FailedPredicateException(this, "precpred(_ctx, 1)");
						setState(60);
						match(OR);
						setState(61);
						expr(2);
						}
						break;
					case 13:
						{
						_localctx = new TermContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(62);
						if (!(precpred(_ctx, 9))) throw new FailedPredicateException(this, "precpred(_ctx, 9)");
						setState(63);
						((TermContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !(_la==IN || _la==NIN) ) {
							((TermContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(64);
						match(T__2);
						setState(65);
						expr(0);
						setState(70);
						_errHandler.sync(this);
						_alt = getInterpreter().adaptivePredict(_input,1,_ctx);
						while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
							if ( _alt==1 ) {
								{
								{
								setState(66);
								match(T__3);
								setState(67);
								expr(0);
								}
								} 
							}
							setState(72);
							_errHandler.sync(this);
							_alt = getInterpreter().adaptivePredict(_input,1,_ctx);
						}
						setState(74);
						_errHandler.sync(this);
						_la = _input.LA(1);
						if (_la==T__3) {
							{
							setState(73);
							match(T__3);
							}
						}

						setState(76);
						match(T__4);
						}
						break;
					}
					} 
				}
				setState(82);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,4,_ctx);
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

	public static class TypeNameContext extends ParserRuleContext {
		public Token ty;
		public TerminalNode BOOL() { return getToken(PlanParser.BOOL, 0); }
		public TerminalNode INT8() { return getToken(PlanParser.INT8, 0); }
		public TerminalNode INT16() { return getToken(PlanParser.INT16, 0); }
		public TerminalNode INT32() { return getToken(PlanParser.INT32, 0); }
		public TerminalNode INT64() { return getToken(PlanParser.INT64, 0); }
		public TerminalNode FLOAT() { return getToken(PlanParser.FLOAT, 0); }
		public TerminalNode DOUBLE() { return getToken(PlanParser.DOUBLE, 0); }
		public TypeNameContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_typeName; }
	}

	public final TypeNameContext typeName() throws RecognitionException {
		TypeNameContext _localctx = new TypeNameContext(_ctx, getState());
		enterRule(_localctx, 2, RULE_typeName);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(83);
			((TypeNameContext)_localctx).ty = _input.LT(1);
			_la = _input.LA(1);
			if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << BOOL) | (1L << INT8) | (1L << INT16) | (1L << INT32) | (1L << INT64) | (1L << FLOAT) | (1L << DOUBLE))) != 0)) ) {
				((TypeNameContext)_localctx).ty = (Token)_errHandler.recoverInline(this);
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
			return expr_sempred((ExprContext)_localctx, predIndex);
		}
		return true;
	}
	private boolean expr_sempred(ExprContext _localctx, int predIndex) {
		switch (predIndex) {
		case 0:
			return precpred(_ctx, 15);
		case 1:
			return precpred(_ctx, 12);
		case 2:
			return precpred(_ctx, 11);
		case 3:
			return precpred(_ctx, 10);
		case 4:
			return precpred(_ctx, 8);
		case 5:
			return precpred(_ctx, 7);
		case 6:
			return precpred(_ctx, 6);
		case 7:
			return precpred(_ctx, 5);
		case 8:
			return precpred(_ctx, 4);
		case 9:
			return precpred(_ctx, 3);
		case 10:
			return precpred(_ctx, 2);
		case 11:
			return precpred(_ctx, 1);
		case 12:
			return precpred(_ctx, 9);
		}
		return true;
	}

	public static final String _serializedATN =
		"\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\3,X\4\2\t\2\4\3\t\3"+
		"\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\5"+
		"\2\30\n\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2"+
		"\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3"+
		"\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\7\2G\n\2\f\2\16\2J"+
		"\13\2\3\2\5\2M\n\2\3\2\3\2\7\2Q\n\2\f\2\16\2T\13\2\3\3\3\3\3\3\2\3\2\4"+
		"\2\4\2\13\4\2\25\26\"#\3\2\27\31\3\2\25\26\3\2\33\34\3\2\17\20\3\2\17"+
		"\22\3\2\23\24\3\2$%\3\2\b\16\2k\2\27\3\2\2\2\4U\3\2\2\2\6\7\b\2\1\2\7"+
		"\30\7\'\2\2\b\30\7(\2\2\t\30\7&\2\2\n\30\7*\2\2\13\30\7)\2\2\f\r\7\3\2"+
		"\2\r\16\5\2\2\2\16\17\7\4\2\2\17\30\3\2\2\2\20\21\t\2\2\2\21\30\5\2\2"+
		"\20\22\23\7\3\2\2\23\24\5\4\3\2\24\25\7\4\2\2\25\26\5\2\2\17\26\30\3\2"+
		"\2\2\27\6\3\2\2\2\27\b\3\2\2\2\27\t\3\2\2\2\27\n\3\2\2\2\27\13\3\2\2\2"+
		"\27\f\3\2\2\2\27\20\3\2\2\2\27\22\3\2\2\2\30R\3\2\2\2\31\32\f\21\2\2\32"+
		"\33\7\32\2\2\33Q\5\2\2\22\34\35\f\16\2\2\35\36\t\3\2\2\36Q\5\2\2\17\37"+
		" \f\r\2\2 !\t\4\2\2!Q\5\2\2\16\"#\f\f\2\2#$\t\5\2\2$Q\5\2\2\r%&\f\n\2"+
		"\2&\'\t\6\2\2\'(\5\2\2\2()\t\6\2\2)*\5\2\2\13*Q\3\2\2\2+,\f\t\2\2,-\t"+
		"\7\2\2-Q\5\2\2\n./\f\b\2\2/\60\t\b\2\2\60Q\5\2\2\t\61\62\f\7\2\2\62\63"+
		"\7\35\2\2\63Q\5\2\2\b\64\65\f\6\2\2\65\66\7\37\2\2\66Q\5\2\2\7\678\f\5"+
		"\2\289\7\36\2\29Q\5\2\2\6:;\f\4\2\2;<\7 \2\2<Q\5\2\2\5=>\f\3\2\2>?\7!"+
		"\2\2?Q\5\2\2\4@A\f\13\2\2AB\t\t\2\2BC\7\5\2\2CH\5\2\2\2DE\7\6\2\2EG\5"+
		"\2\2\2FD\3\2\2\2GJ\3\2\2\2HF\3\2\2\2HI\3\2\2\2IL\3\2\2\2JH\3\2\2\2KM\7"+
		"\6\2\2LK\3\2\2\2LM\3\2\2\2MN\3\2\2\2NO\7\7\2\2OQ\3\2\2\2P\31\3\2\2\2P"+
		"\34\3\2\2\2P\37\3\2\2\2P\"\3\2\2\2P%\3\2\2\2P+\3\2\2\2P.\3\2\2\2P\61\3"+
		"\2\2\2P\64\3\2\2\2P\67\3\2\2\2P:\3\2\2\2P=\3\2\2\2P@\3\2\2\2QT\3\2\2\2"+
		"RP\3\2\2\2RS\3\2\2\2S\3\3\2\2\2TR\3\2\2\2UV\t\n\2\2V\5\3\2\2\2\7\27HL"+
		"PR";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}