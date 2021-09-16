// Generated from /home/xa/milvus/internal/proxy/plan_parser/Plan.g4 by ANTLR 4.8
import org.antlr.v4.runtime.Lexer;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.TokenStream;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.misc.*;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast"})
public class PlanLexer extends Lexer {
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
	public static String[] channelNames = {
		"DEFAULT_TOKEN_CHANNEL", "HIDDEN"
	};

	public static String[] modeNames = {
		"DEFAULT_MODE"
	};

	private static String[] makeRuleNames() {
		return new String[] {
			"T__0", "T__1", "T__2", "T__3", "T__4", "BOOL", "INT8", "INT16", "INT32", 
			"INT64", "FLOAT", "DOUBLE", "LT", "LE", "GT", "GE", "EQ", "NE", "ADD", 
			"SUB", "MUL", "DIV", "MOD", "POW", "SHL", "SHR", "BAND", "BOR", "BXOR", 
			"AND", "OR", "BNOT", "NOT", "IN", "NIN", "BooleanConstant", "IntegerConstant", 
			"FloatingConstant", "Identifier", "StringLiteral", "EncodingPrefix", 
			"SCharSequence", "SChar", "Nondigit", "Digit", "BinaryConstant", "DecimalConstant", 
			"OctalConstant", "HexadecimalConstant", "NonzeroDigit", "OctalDigit", 
			"HexadecimalDigit", "HexQuad", "UniversalCharacterName", "DecimalFloatingConstant", 
			"HexadecimalFloatingConstant", "FractionalConstant", "ExponentPart", 
			"DigitSequence", "HexadecimalFractionalConstant", "HexadecimalDigitSequence", 
			"BinaryExponentPart", "EscapeSequence", "Whitespace", "Newline"
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


	public PlanLexer(CharStream input) {
		super(input);
		_interp = new LexerATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	@Override
	public String getGrammarFileName() { return "Plan.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public String[] getChannelNames() { return channelNames; }

	@Override
	public String[] getModeNames() { return modeNames; }

	@Override
	public ATN getATN() { return _ATN; }

	public static final String _serializedATN =
		"\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\2,\u01c9\b\1\4\2\t"+
		"\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4\13"+
		"\t\13\4\f\t\f\4\r\t\r\4\16\t\16\4\17\t\17\4\20\t\20\4\21\t\21\4\22\t\22"+
		"\4\23\t\23\4\24\t\24\4\25\t\25\4\26\t\26\4\27\t\27\4\30\t\30\4\31\t\31"+
		"\4\32\t\32\4\33\t\33\4\34\t\34\4\35\t\35\4\36\t\36\4\37\t\37\4 \t \4!"+
		"\t!\4\"\t\"\4#\t#\4$\t$\4%\t%\4&\t&\4\'\t\'\4(\t(\4)\t)\4*\t*\4+\t+\4"+
		",\t,\4-\t-\4.\t.\4/\t/\4\60\t\60\4\61\t\61\4\62\t\62\4\63\t\63\4\64\t"+
		"\64\4\65\t\65\4\66\t\66\4\67\t\67\48\t8\49\t9\4:\t:\4;\t;\4<\t<\4=\t="+
		"\4>\t>\4?\t?\4@\t@\4A\tA\4B\tB\3\2\3\2\3\3\3\3\3\4\3\4\3\5\3\5\3\6\3\6"+
		"\3\7\3\7\3\7\3\7\3\7\3\b\3\b\3\b\3\b\3\b\3\t\3\t\3\t\3\t\3\t\3\t\3\n\3"+
		"\n\3\n\3\n\3\n\3\n\3\13\3\13\3\13\3\13\3\13\3\13\3\f\3\f\3\f\3\f\3\f\3"+
		"\f\3\r\3\r\3\r\3\r\3\r\3\r\3\r\3\16\3\16\3\17\3\17\3\17\3\20\3\20\3\21"+
		"\3\21\3\21\3\22\3\22\3\22\3\23\3\23\3\23\3\24\3\24\3\25\3\25\3\26\3\26"+
		"\3\27\3\27\3\30\3\30\3\31\3\31\3\31\3\32\3\32\3\32\3\33\3\33\3\33\3\34"+
		"\3\34\3\35\3\35\3\36\3\36\3\37\3\37\3\37\3\37\3\37\5\37\u00e7\n\37\3 "+
		"\3 \3 \3 \5 \u00ed\n \3!\3!\3\"\3\"\3\"\3\"\5\"\u00f5\n\"\3#\3#\3#\3$"+
		"\3$\3$\3$\3$\3$\3$\3%\3%\3%\3%\3%\3%\3%\3%\3%\5%\u010a\n%\3&\3&\3&\3&"+
		"\5&\u0110\n&\3\'\3\'\5\'\u0114\n\'\3(\3(\3(\7(\u0119\n(\f(\16(\u011c\13"+
		"(\3)\5)\u011f\n)\3)\3)\5)\u0123\n)\3)\3)\3*\3*\3*\5*\u012a\n*\3+\6+\u012d"+
		"\n+\r+\16+\u012e\3,\3,\3,\3,\3,\3,\3,\5,\u0138\n,\3-\3-\3.\3.\3/\3/\3"+
		"/\6/\u0141\n/\r/\16/\u0142\3\60\3\60\7\60\u0147\n\60\f\60\16\60\u014a"+
		"\13\60\3\61\3\61\7\61\u014e\n\61\f\61\16\61\u0151\13\61\3\62\3\62\3\62"+
		"\3\62\3\63\3\63\3\64\3\64\3\65\3\65\3\66\3\66\3\66\3\66\3\66\3\67\3\67"+
		"\3\67\3\67\3\67\3\67\3\67\3\67\3\67\3\67\5\67\u016c\n\67\38\38\58\u0170"+
		"\n8\38\38\38\58\u0175\n8\39\39\39\39\59\u017b\n9\39\39\3:\5:\u0180\n:"+
		"\3:\3:\3:\3:\3:\5:\u0187\n:\3;\3;\5;\u018b\n;\3;\3;\3<\6<\u0190\n<\r<"+
		"\16<\u0191\3=\5=\u0195\n=\3=\3=\3=\3=\3=\5=\u019c\n=\3>\6>\u019f\n>\r"+
		">\16>\u01a0\3?\3?\5?\u01a5\n?\3?\3?\3@\3@\3@\3@\3@\5@\u01ae\n@\3@\5@\u01b1"+
		"\n@\3@\3@\3@\3@\3@\5@\u01b8\n@\3A\6A\u01bb\nA\rA\16A\u01bc\3A\3A\3B\3"+
		"B\5B\u01c3\nB\3B\5B\u01c6\nB\3B\3B\2\2C\3\3\5\4\7\5\t\6\13\7\r\b\17\t"+
		"\21\n\23\13\25\f\27\r\31\16\33\17\35\20\37\21!\22#\23%\24\'\25)\26+\27"+
		"-\30/\31\61\32\63\33\65\34\67\359\36;\37= ?!A\"C#E$G%I&K\'M(O)Q*S\2U\2"+
		"W\2Y\2[\2]\2_\2a\2c\2e\2g\2i\2k\2m\2o\2q\2s\2u\2w\2y\2{\2}\2\177\2\u0081"+
		"+\u0083,\3\2\21\5\2NNWWww\6\2\f\f\17\17$$^^\5\2C\\aac|\3\2\62;\4\2DDd"+
		"d\3\2\62\63\4\2ZZzz\3\2\63;\3\2\629\5\2\62;CHch\4\2GGgg\4\2--//\4\2RR"+
		"rr\f\2$$))AA^^cdhhppttvvxx\4\2\13\13\"\"\2\u01d9\2\3\3\2\2\2\2\5\3\2\2"+
		"\2\2\7\3\2\2\2\2\t\3\2\2\2\2\13\3\2\2\2\2\r\3\2\2\2\2\17\3\2\2\2\2\21"+
		"\3\2\2\2\2\23\3\2\2\2\2\25\3\2\2\2\2\27\3\2\2\2\2\31\3\2\2\2\2\33\3\2"+
		"\2\2\2\35\3\2\2\2\2\37\3\2\2\2\2!\3\2\2\2\2#\3\2\2\2\2%\3\2\2\2\2\'\3"+
		"\2\2\2\2)\3\2\2\2\2+\3\2\2\2\2-\3\2\2\2\2/\3\2\2\2\2\61\3\2\2\2\2\63\3"+
		"\2\2\2\2\65\3\2\2\2\2\67\3\2\2\2\29\3\2\2\2\2;\3\2\2\2\2=\3\2\2\2\2?\3"+
		"\2\2\2\2A\3\2\2\2\2C\3\2\2\2\2E\3\2\2\2\2G\3\2\2\2\2I\3\2\2\2\2K\3\2\2"+
		"\2\2M\3\2\2\2\2O\3\2\2\2\2Q\3\2\2\2\2\u0081\3\2\2\2\2\u0083\3\2\2\2\3"+
		"\u0085\3\2\2\2\5\u0087\3\2\2\2\7\u0089\3\2\2\2\t\u008b\3\2\2\2\13\u008d"+
		"\3\2\2\2\r\u008f\3\2\2\2\17\u0094\3\2\2\2\21\u0099\3\2\2\2\23\u009f\3"+
		"\2\2\2\25\u00a5\3\2\2\2\27\u00ab\3\2\2\2\31\u00b1\3\2\2\2\33\u00b8\3\2"+
		"\2\2\35\u00ba\3\2\2\2\37\u00bd\3\2\2\2!\u00bf\3\2\2\2#\u00c2\3\2\2\2%"+
		"\u00c5\3\2\2\2\'\u00c8\3\2\2\2)\u00ca\3\2\2\2+\u00cc\3\2\2\2-\u00ce\3"+
		"\2\2\2/\u00d0\3\2\2\2\61\u00d2\3\2\2\2\63\u00d5\3\2\2\2\65\u00d8\3\2\2"+
		"\2\67\u00db\3\2\2\29\u00dd\3\2\2\2;\u00df\3\2\2\2=\u00e6\3\2\2\2?\u00ec"+
		"\3\2\2\2A\u00ee\3\2\2\2C\u00f4\3\2\2\2E\u00f6\3\2\2\2G\u00f9\3\2\2\2I"+
		"\u0109\3\2\2\2K\u010f\3\2\2\2M\u0113\3\2\2\2O\u0115\3\2\2\2Q\u011e\3\2"+
		"\2\2S\u0129\3\2\2\2U\u012c\3\2\2\2W\u0137\3\2\2\2Y\u0139\3\2\2\2[\u013b"+
		"\3\2\2\2]\u013d\3\2\2\2_\u0144\3\2\2\2a\u014b\3\2\2\2c\u0152\3\2\2\2e"+
		"\u0156\3\2\2\2g\u0158\3\2\2\2i\u015a\3\2\2\2k\u015c\3\2\2\2m\u016b\3\2"+
		"\2\2o\u0174\3\2\2\2q\u0176\3\2\2\2s\u0186\3\2\2\2u\u0188\3\2\2\2w\u018f"+
		"\3\2\2\2y\u019b\3\2\2\2{\u019e\3\2\2\2}\u01a2\3\2\2\2\177\u01b7\3\2\2"+
		"\2\u0081\u01ba\3\2\2\2\u0083\u01c5\3\2\2\2\u0085\u0086\7*\2\2\u0086\4"+
		"\3\2\2\2\u0087\u0088\7+\2\2\u0088\6\3\2\2\2\u0089\u008a\7]\2\2\u008a\b"+
		"\3\2\2\2\u008b\u008c\7.\2\2\u008c\n\3\2\2\2\u008d\u008e\7_\2\2\u008e\f"+
		"\3\2\2\2\u008f\u0090\7d\2\2\u0090\u0091\7q\2\2\u0091\u0092\7q\2\2\u0092"+
		"\u0093\7n\2\2\u0093\16\3\2\2\2\u0094\u0095\7k\2\2\u0095\u0096\7p\2\2\u0096"+
		"\u0097\7v\2\2\u0097\u0098\7:\2\2\u0098\20\3\2\2\2\u0099\u009a\7k\2\2\u009a"+
		"\u009b\7p\2\2\u009b\u009c\7v\2\2\u009c\u009d\7\63\2\2\u009d\u009e\78\2"+
		"\2\u009e\22\3\2\2\2\u009f\u00a0\7k\2\2\u00a0\u00a1\7p\2\2\u00a1\u00a2"+
		"\7v\2\2\u00a2\u00a3\7\65\2\2\u00a3\u00a4\7\64\2\2\u00a4\24\3\2\2\2\u00a5"+
		"\u00a6\7k\2\2\u00a6\u00a7\7p\2\2\u00a7\u00a8\7v\2\2\u00a8\u00a9\78\2\2"+
		"\u00a9\u00aa\7\66\2\2\u00aa\26\3\2\2\2\u00ab\u00ac\7h\2\2\u00ac\u00ad"+
		"\7n\2\2\u00ad\u00ae\7q\2\2\u00ae\u00af\7c\2\2\u00af\u00b0\7v\2\2\u00b0"+
		"\30\3\2\2\2\u00b1\u00b2\7f\2\2\u00b2\u00b3\7q\2\2\u00b3\u00b4\7w\2\2\u00b4"+
		"\u00b5\7d\2\2\u00b5\u00b6\7n\2\2\u00b6\u00b7\7g\2\2\u00b7\32\3\2\2\2\u00b8"+
		"\u00b9\7>\2\2\u00b9\34\3\2\2\2\u00ba\u00bb\7>\2\2\u00bb\u00bc\7?\2\2\u00bc"+
		"\36\3\2\2\2\u00bd\u00be\7@\2\2\u00be \3\2\2\2\u00bf\u00c0\7@\2\2\u00c0"+
		"\u00c1\7?\2\2\u00c1\"\3\2\2\2\u00c2\u00c3\7?\2\2\u00c3\u00c4\7?\2\2\u00c4"+
		"$\3\2\2\2\u00c5\u00c6\7#\2\2\u00c6\u00c7\7?\2\2\u00c7&\3\2\2\2\u00c8\u00c9"+
		"\7-\2\2\u00c9(\3\2\2\2\u00ca\u00cb\7/\2\2\u00cb*\3\2\2\2\u00cc\u00cd\7"+
		",\2\2\u00cd,\3\2\2\2\u00ce\u00cf\7\61\2\2\u00cf.\3\2\2\2\u00d0\u00d1\7"+
		"\'\2\2\u00d1\60\3\2\2\2\u00d2\u00d3\7,\2\2\u00d3\u00d4\7,\2\2\u00d4\62"+
		"\3\2\2\2\u00d5\u00d6\7>\2\2\u00d6\u00d7\7>\2\2\u00d7\64\3\2\2\2\u00d8"+
		"\u00d9\7@\2\2\u00d9\u00da\7@\2\2\u00da\66\3\2\2\2\u00db\u00dc\7(\2\2\u00dc"+
		"8\3\2\2\2\u00dd\u00de\7~\2\2\u00de:\3\2\2\2\u00df\u00e0\7`\2\2\u00e0<"+
		"\3\2\2\2\u00e1\u00e2\7(\2\2\u00e2\u00e7\7(\2\2\u00e3\u00e4\7c\2\2\u00e4"+
		"\u00e5\7p\2\2\u00e5\u00e7\7f\2\2\u00e6\u00e1\3\2\2\2\u00e6\u00e3\3\2\2"+
		"\2\u00e7>\3\2\2\2\u00e8\u00e9\7~\2\2\u00e9\u00ed\7~\2\2\u00ea\u00eb\7"+
		"q\2\2\u00eb\u00ed\7t\2\2\u00ec\u00e8\3\2\2\2\u00ec\u00ea\3\2\2\2\u00ed"+
		"@\3\2\2\2\u00ee\u00ef\7\u0080\2\2\u00efB\3\2\2\2\u00f0\u00f5\7#\2\2\u00f1"+
		"\u00f2\7p\2\2\u00f2\u00f3\7q\2\2\u00f3\u00f5\7v\2\2\u00f4\u00f0\3\2\2"+
		"\2\u00f4\u00f1\3\2\2\2\u00f5D\3\2\2\2\u00f6\u00f7\7k\2\2\u00f7\u00f8\7"+
		"p\2\2\u00f8F\3\2\2\2\u00f9\u00fa\7p\2\2\u00fa\u00fb\7q\2\2\u00fb\u00fc"+
		"\7v\2\2\u00fc\u00fd\7\"\2\2\u00fd\u00fe\7k\2\2\u00fe\u00ff\7p\2\2\u00ff"+
		"H\3\2\2\2\u0100\u0101\7v\2\2\u0101\u0102\7t\2\2\u0102\u0103\7w\2\2\u0103"+
		"\u010a\7g\2\2\u0104\u0105\7h\2\2\u0105\u0106\7c\2\2\u0106\u0107\7n\2\2"+
		"\u0107\u0108\7u\2\2\u0108\u010a\7g\2\2\u0109\u0100\3\2\2\2\u0109\u0104"+
		"\3\2\2\2\u010aJ\3\2\2\2\u010b\u0110\5_\60\2\u010c\u0110\5a\61\2\u010d"+
		"\u0110\5c\62\2\u010e\u0110\5]/\2\u010f\u010b\3\2\2\2\u010f\u010c\3\2\2"+
		"\2\u010f\u010d\3\2\2\2\u010f\u010e\3\2\2\2\u0110L\3\2\2\2\u0111\u0114"+
		"\5o8\2\u0112\u0114\5q9\2\u0113\u0111\3\2\2\2\u0113\u0112\3\2\2\2\u0114"+
		"N\3\2\2\2\u0115\u011a\5Y-\2\u0116\u0119\5Y-\2\u0117\u0119\5[.\2\u0118"+
		"\u0116\3\2\2\2\u0118\u0117\3\2\2\2\u0119\u011c\3\2\2\2\u011a\u0118\3\2"+
		"\2\2\u011a\u011b\3\2\2\2\u011bP\3\2\2\2\u011c\u011a\3\2\2\2\u011d\u011f"+
		"\5S*\2\u011e\u011d\3\2\2\2\u011e\u011f\3\2\2\2\u011f\u0120\3\2\2\2\u0120"+
		"\u0122\7$\2\2\u0121\u0123\5U+\2\u0122\u0121\3\2\2\2\u0122\u0123\3\2\2"+
		"\2\u0123\u0124\3\2\2\2\u0124\u0125\7$\2\2\u0125R\3\2\2\2\u0126\u0127\7"+
		"w\2\2\u0127\u012a\7:\2\2\u0128\u012a\t\2\2\2\u0129\u0126\3\2\2\2\u0129"+
		"\u0128\3\2\2\2\u012aT\3\2\2\2\u012b\u012d\5W,\2\u012c\u012b\3\2\2\2\u012d"+
		"\u012e\3\2\2\2\u012e\u012c\3\2\2\2\u012e\u012f\3\2\2\2\u012fV\3\2\2\2"+
		"\u0130\u0138\n\3\2\2\u0131\u0138\5\177@\2\u0132\u0133\7^\2\2\u0133\u0138"+
		"\7\f\2\2\u0134\u0135\7^\2\2\u0135\u0136\7\17\2\2\u0136\u0138\7\f\2\2\u0137"+
		"\u0130\3\2\2\2\u0137\u0131\3\2\2\2\u0137\u0132\3\2\2\2\u0137\u0134\3\2"+
		"\2\2\u0138X\3\2\2\2\u0139\u013a\t\4\2\2\u013aZ\3\2\2\2\u013b\u013c\t\5"+
		"\2\2\u013c\\\3\2\2\2\u013d\u013e\7\62\2\2\u013e\u0140\t\6\2\2\u013f\u0141"+
		"\t\7\2\2\u0140\u013f\3\2\2\2\u0141\u0142\3\2\2\2\u0142\u0140\3\2\2\2\u0142"+
		"\u0143\3\2\2\2\u0143^\3\2\2\2\u0144\u0148\5e\63\2\u0145\u0147\5[.\2\u0146"+
		"\u0145\3\2\2\2\u0147\u014a\3\2\2\2\u0148\u0146\3\2\2\2\u0148\u0149\3\2"+
		"\2\2\u0149`\3\2\2\2\u014a\u0148\3\2\2\2\u014b\u014f\7\62\2\2\u014c\u014e"+
		"\5g\64\2\u014d\u014c\3\2\2\2\u014e\u0151\3\2\2\2\u014f\u014d\3\2\2\2\u014f"+
		"\u0150\3\2\2\2\u0150b\3\2\2\2\u0151\u014f\3\2\2\2\u0152\u0153\7\62\2\2"+
		"\u0153\u0154\t\b\2\2\u0154\u0155\5{>\2\u0155d\3\2\2\2\u0156\u0157\t\t"+
		"\2\2\u0157f\3\2\2\2\u0158\u0159\t\n\2\2\u0159h\3\2\2\2\u015a\u015b\t\13"+
		"\2\2\u015bj\3\2\2\2\u015c\u015d\5i\65\2\u015d\u015e\5i\65\2\u015e\u015f"+
		"\5i\65\2\u015f\u0160\5i\65\2\u0160l\3\2\2\2\u0161\u0162\7^\2\2\u0162\u0163"+
		"\7w\2\2\u0163\u0164\3\2\2\2\u0164\u016c\5k\66\2\u0165\u0166\7^\2\2\u0166"+
		"\u0167\7W\2\2\u0167\u0168\3\2\2\2\u0168\u0169\5k\66\2\u0169\u016a\5k\66"+
		"\2\u016a\u016c\3\2\2\2\u016b\u0161\3\2\2\2\u016b\u0165\3\2\2\2\u016cn"+
		"\3\2\2\2\u016d\u016f\5s:\2\u016e\u0170\5u;\2\u016f\u016e\3\2\2\2\u016f"+
		"\u0170\3\2\2\2\u0170\u0175\3\2\2\2\u0171\u0172\5w<\2\u0172\u0173\5u;\2"+
		"\u0173\u0175\3\2\2\2\u0174\u016d\3\2\2\2\u0174\u0171\3\2\2\2\u0175p\3"+
		"\2\2\2\u0176\u0177\7\62\2\2\u0177\u017a\t\b\2\2\u0178\u017b\5y=\2\u0179"+
		"\u017b\5{>\2\u017a\u0178\3\2\2\2\u017a\u0179\3\2\2\2\u017b\u017c\3\2\2"+
		"\2\u017c\u017d\5}?\2\u017dr\3\2\2\2\u017e\u0180\5w<\2\u017f\u017e\3\2"+
		"\2\2\u017f\u0180\3\2\2\2\u0180\u0181\3\2\2\2\u0181\u0182\7\60\2\2\u0182"+
		"\u0187\5w<\2\u0183\u0184\5w<\2\u0184\u0185\7\60\2\2\u0185\u0187\3\2\2"+
		"\2\u0186\u017f\3\2\2\2\u0186\u0183\3\2\2\2\u0187t\3\2\2\2\u0188\u018a"+
		"\t\f\2\2\u0189\u018b\t\r\2\2\u018a\u0189\3\2\2\2\u018a\u018b\3\2\2\2\u018b"+
		"\u018c\3\2\2\2\u018c\u018d\5w<\2\u018dv\3\2\2\2\u018e\u0190\5[.\2\u018f"+
		"\u018e\3\2\2\2\u0190\u0191\3\2\2\2\u0191\u018f\3\2\2\2\u0191\u0192\3\2"+
		"\2\2\u0192x\3\2\2\2\u0193\u0195\5{>\2\u0194\u0193\3\2\2\2\u0194\u0195"+
		"\3\2\2\2\u0195\u0196\3\2\2\2\u0196\u0197\7\60\2\2\u0197\u019c\5{>\2\u0198"+
		"\u0199\5{>\2\u0199\u019a\7\60\2\2\u019a\u019c\3\2\2\2\u019b\u0194\3\2"+
		"\2\2\u019b\u0198\3\2\2\2\u019cz\3\2\2\2\u019d\u019f\5i\65\2\u019e\u019d"+
		"\3\2\2\2\u019f\u01a0\3\2\2\2\u01a0\u019e\3\2\2\2\u01a0\u01a1\3\2\2\2\u01a1"+
		"|\3\2\2\2\u01a2\u01a4\t\16\2\2\u01a3\u01a5\t\r\2\2\u01a4\u01a3\3\2\2\2"+
		"\u01a4\u01a5\3\2\2\2\u01a5\u01a6\3\2\2\2\u01a6\u01a7\5w<\2\u01a7~\3\2"+
		"\2\2\u01a8\u01a9\7^\2\2\u01a9\u01b8\t\17\2\2\u01aa\u01ab\7^\2\2\u01ab"+
		"\u01ad\5g\64\2\u01ac\u01ae\5g\64\2\u01ad\u01ac\3\2\2\2\u01ad\u01ae\3\2"+
		"\2\2\u01ae\u01b0\3\2\2\2\u01af\u01b1\5g\64\2\u01b0\u01af\3\2\2\2\u01b0"+
		"\u01b1\3\2\2\2\u01b1\u01b8\3\2\2\2\u01b2\u01b3\7^\2\2\u01b3\u01b4\7z\2"+
		"\2\u01b4\u01b5\3\2\2\2\u01b5\u01b8\5{>\2\u01b6\u01b8\5m\67\2\u01b7\u01a8"+
		"\3\2\2\2\u01b7\u01aa\3\2\2\2\u01b7\u01b2\3\2\2\2\u01b7\u01b6\3\2\2\2\u01b8"+
		"\u0080\3\2\2\2\u01b9\u01bb\t\20\2\2\u01ba\u01b9\3\2\2\2\u01bb\u01bc\3"+
		"\2\2\2\u01bc\u01ba\3\2\2\2\u01bc\u01bd\3\2\2\2\u01bd\u01be\3\2\2\2\u01be"+
		"\u01bf\bA\2\2\u01bf\u0082\3\2\2\2\u01c0\u01c2\7\17\2\2\u01c1\u01c3\7\f"+
		"\2\2\u01c2\u01c1\3\2\2\2\u01c2\u01c3\3\2\2\2\u01c3\u01c6\3\2\2\2\u01c4"+
		"\u01c6\7\f\2\2\u01c5\u01c0\3\2\2\2\u01c5\u01c4\3\2\2\2\u01c6\u01c7\3\2"+
		"\2\2\u01c7\u01c8\bB\2\2\u01c8\u0084\3\2\2\2%\2\u00e6\u00ec\u00f4\u0109"+
		"\u010f\u0113\u0118\u011a\u011e\u0122\u0129\u012e\u0137\u0142\u0148\u014f"+
		"\u016b\u016f\u0174\u017a\u017f\u0186\u018a\u0191\u0194\u019b\u01a0\u01a4"+
		"\u01ad\u01b0\u01b7\u01bc\u01c2\u01c5\3\b\2\2";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}