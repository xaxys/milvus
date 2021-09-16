// Generated from /home/xa/milvus/internal/proxy/plan_parser/plan.g4 by ANTLR 4.8
import org.antlr.v4.runtime.Lexer;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.TokenStream;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.misc.*;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast"})
public class planLexer extends Lexer {
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
			"SUB", "MUL", "DIV", "MOD", "SHL", "SHR", "BAND", "BOR", "BXOR", "AND", 
			"OR", "BNOT", "NOT", "IN", "TRUE", "FALSE", "Identifier", "StringLiteral", 
			"IntegerConstant", "FloatingConstant", "BooleanConstant", "EncodingPrefix", 
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


	public planLexer(CharStream input) {
		super(input);
		_interp = new LexerATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	@Override
	public String getGrammarFileName() { return "plan.g4"; }

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
		"\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\2,\u01b8\b\1\4\2\t"+
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
		"\3\27\3\27\3\30\3\30\3\31\3\31\3\31\3\32\3\32\3\32\3\33\3\33\3\34\3\34"+
		"\3\35\3\35\3\36\3\36\3\36\3\37\3\37\3\37\3 \3 \3!\3!\3\"\3\"\3\"\3#\3"+
		"#\3#\3#\3#\3$\3$\3$\3$\3$\3$\3%\3%\3%\7%\u00fa\n%\f%\16%\u00fd\13%\3&"+
		"\5&\u0100\n&\3&\3&\5&\u0104\n&\3&\3&\3\'\3\'\3\'\3\'\5\'\u010c\n\'\3("+
		"\3(\5(\u0110\n(\3)\3)\5)\u0114\n)\3*\3*\3*\5*\u0119\n*\3+\6+\u011c\n+"+
		"\r+\16+\u011d\3,\3,\3,\3,\3,\3,\3,\5,\u0127\n,\3-\3-\3.\3.\3/\3/\3/\6"+
		"/\u0130\n/\r/\16/\u0131\3\60\3\60\7\60\u0136\n\60\f\60\16\60\u0139\13"+
		"\60\3\61\3\61\7\61\u013d\n\61\f\61\16\61\u0140\13\61\3\62\3\62\3\62\3"+
		"\62\3\63\3\63\3\64\3\64\3\65\3\65\3\66\3\66\3\66\3\66\3\66\3\67\3\67\3"+
		"\67\3\67\3\67\3\67\3\67\3\67\3\67\3\67\5\67\u015b\n\67\38\38\58\u015f"+
		"\n8\38\38\38\58\u0164\n8\39\39\39\39\59\u016a\n9\39\39\3:\5:\u016f\n:"+
		"\3:\3:\3:\3:\3:\5:\u0176\n:\3;\3;\5;\u017a\n;\3;\3;\3<\6<\u017f\n<\r<"+
		"\16<\u0180\3=\5=\u0184\n=\3=\3=\3=\3=\3=\5=\u018b\n=\3>\6>\u018e\n>\r"+
		">\16>\u018f\3?\3?\5?\u0194\n?\3?\3?\3@\3@\3@\3@\3@\5@\u019d\n@\3@\5@\u01a0"+
		"\n@\3@\3@\3@\3@\3@\5@\u01a7\n@\3A\6A\u01aa\nA\rA\16A\u01ab\3A\3A\3B\3"+
		"B\5B\u01b2\nB\3B\5B\u01b5\nB\3B\3B\2\2C\3\3\5\4\7\5\t\6\13\7\r\b\17\t"+
		"\21\n\23\13\25\f\27\r\31\16\33\17\35\20\37\21!\22#\23%\24\'\25)\26+\27"+
		"-\30/\31\61\32\63\33\65\34\67\359\36;\37= ?!A\"C#E$G%I&K\'M(O)Q*S\2U\2"+
		"W\2Y\2[\2]\2_\2a\2c\2e\2g\2i\2k\2m\2o\2q\2s\2u\2w\2y\2{\2}\2\177\2\u0081"+
		"+\u0083,\3\2\21\5\2NNWWww\6\2\f\f\17\17$$^^\5\2C\\aac|\3\2\62;\4\2DDd"+
		"d\3\2\62\63\4\2ZZzz\3\2\63;\3\2\629\5\2\62;CHch\4\2GGgg\4\2--//\4\2RR"+
		"rr\f\2$$))AA^^cdhhppttvvxx\4\2\13\13\"\"\2\u01c5\2\3\3\2\2\2\2\5\3\2\2"+
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
		"\2\67\u00da\3\2\2\29\u00dc\3\2\2\2;\u00de\3\2\2\2=\u00e1\3\2\2\2?\u00e4"+
		"\3\2\2\2A\u00e6\3\2\2\2C\u00e8\3\2\2\2E\u00eb\3\2\2\2G\u00f0\3\2\2\2I"+
		"\u00f6\3\2\2\2K\u00ff\3\2\2\2M\u010b\3\2\2\2O\u010f\3\2\2\2Q\u0113\3\2"+
		"\2\2S\u0118\3\2\2\2U\u011b\3\2\2\2W\u0126\3\2\2\2Y\u0128\3\2\2\2[\u012a"+
		"\3\2\2\2]\u012c\3\2\2\2_\u0133\3\2\2\2a\u013a\3\2\2\2c\u0141\3\2\2\2e"+
		"\u0145\3\2\2\2g\u0147\3\2\2\2i\u0149\3\2\2\2k\u014b\3\2\2\2m\u015a\3\2"+
		"\2\2o\u0163\3\2\2\2q\u0165\3\2\2\2s\u0175\3\2\2\2u\u0177\3\2\2\2w\u017e"+
		"\3\2\2\2y\u018a\3\2\2\2{\u018d\3\2\2\2}\u0191\3\2\2\2\177\u01a6\3\2\2"+
		"\2\u0081\u01a9\3\2\2\2\u0083\u01b4\3\2\2\2\u0085\u0086\7*\2\2\u0086\4"+
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
		"\'\2\2\u00d1\60\3\2\2\2\u00d2\u00d3\7>\2\2\u00d3\u00d4\7>\2\2\u00d4\62"+
		"\3\2\2\2\u00d5\u00d6\7@\2\2\u00d6\u00d7\7@\2\2\u00d7\64\3\2\2\2\u00d8"+
		"\u00d9\7(\2\2\u00d9\66\3\2\2\2\u00da\u00db\7~\2\2\u00db8\3\2\2\2\u00dc"+
		"\u00dd\7`\2\2\u00dd:\3\2\2\2\u00de\u00df\7(\2\2\u00df\u00e0\7(\2\2\u00e0"+
		"<\3\2\2\2\u00e1\u00e2\7~\2\2\u00e2\u00e3\7~\2\2\u00e3>\3\2\2\2\u00e4\u00e5"+
		"\7\u0080\2\2\u00e5@\3\2\2\2\u00e6\u00e7\7#\2\2\u00e7B\3\2\2\2\u00e8\u00e9"+
		"\7k\2\2\u00e9\u00ea\7p\2\2\u00eaD\3\2\2\2\u00eb\u00ec\7v\2\2\u00ec\u00ed"+
		"\7t\2\2\u00ed\u00ee\7w\2\2\u00ee\u00ef\7g\2\2\u00efF\3\2\2\2\u00f0\u00f1"+
		"\7h\2\2\u00f1\u00f2\7c\2\2\u00f2\u00f3\7n\2\2\u00f3\u00f4\7u\2\2\u00f4"+
		"\u00f5\7g\2\2\u00f5H\3\2\2\2\u00f6\u00fb\5Y-\2\u00f7\u00fa\5Y-\2\u00f8"+
		"\u00fa\5[.\2\u00f9\u00f7\3\2\2\2\u00f9\u00f8\3\2\2\2\u00fa\u00fd\3\2\2"+
		"\2\u00fb\u00f9\3\2\2\2\u00fb\u00fc\3\2\2\2\u00fcJ\3\2\2\2\u00fd\u00fb"+
		"\3\2\2\2\u00fe\u0100\5S*\2\u00ff\u00fe\3\2\2\2\u00ff\u0100\3\2\2\2\u0100"+
		"\u0101\3\2\2\2\u0101\u0103\7$\2\2\u0102\u0104\5U+\2\u0103\u0102\3\2\2"+
		"\2\u0103\u0104\3\2\2\2\u0104\u0105\3\2\2\2\u0105\u0106\7$\2\2\u0106L\3"+
		"\2\2\2\u0107\u010c\5_\60\2\u0108\u010c\5a\61\2\u0109\u010c\5c\62\2\u010a"+
		"\u010c\5]/\2\u010b\u0107\3\2\2\2\u010b\u0108\3\2\2\2\u010b\u0109\3\2\2"+
		"\2\u010b\u010a\3\2\2\2\u010cN\3\2\2\2\u010d\u0110\5o8\2\u010e\u0110\5"+
		"q9\2\u010f\u010d\3\2\2\2\u010f\u010e\3\2\2\2\u0110P\3\2\2\2\u0111\u0114"+
		"\5E#\2\u0112\u0114\5G$\2\u0113\u0111\3\2\2\2\u0113\u0112\3\2\2\2\u0114"+
		"R\3\2\2\2\u0115\u0116\7w\2\2\u0116\u0119\7:\2\2\u0117\u0119\t\2\2\2\u0118"+
		"\u0115\3\2\2\2\u0118\u0117\3\2\2\2\u0119T\3\2\2\2\u011a\u011c\5W,\2\u011b"+
		"\u011a\3\2\2\2\u011c\u011d\3\2\2\2\u011d\u011b\3\2\2\2\u011d\u011e\3\2"+
		"\2\2\u011eV\3\2\2\2\u011f\u0127\n\3\2\2\u0120\u0127\5\177@\2\u0121\u0122"+
		"\7^\2\2\u0122\u0127\7\f\2\2\u0123\u0124\7^\2\2\u0124\u0125\7\17\2\2\u0125"+
		"\u0127\7\f\2\2\u0126\u011f\3\2\2\2\u0126\u0120\3\2\2\2\u0126\u0121\3\2"+
		"\2\2\u0126\u0123\3\2\2\2\u0127X\3\2\2\2\u0128\u0129\t\4\2\2\u0129Z\3\2"+
		"\2\2\u012a\u012b\t\5\2\2\u012b\\\3\2\2\2\u012c\u012d\7\62\2\2\u012d\u012f"+
		"\t\6\2\2\u012e\u0130\t\7\2\2\u012f\u012e\3\2\2\2\u0130\u0131\3\2\2\2\u0131"+
		"\u012f\3\2\2\2\u0131\u0132\3\2\2\2\u0132^\3\2\2\2\u0133\u0137\5e\63\2"+
		"\u0134\u0136\5[.\2\u0135\u0134\3\2\2\2\u0136\u0139\3\2\2\2\u0137\u0135"+
		"\3\2\2\2\u0137\u0138\3\2\2\2\u0138`\3\2\2\2\u0139\u0137\3\2\2\2\u013a"+
		"\u013e\7\62\2\2\u013b\u013d\5g\64\2\u013c\u013b\3\2\2\2\u013d\u0140\3"+
		"\2\2\2\u013e\u013c\3\2\2\2\u013e\u013f\3\2\2\2\u013fb\3\2\2\2\u0140\u013e"+
		"\3\2\2\2\u0141\u0142\7\62\2\2\u0142\u0143\t\b\2\2\u0143\u0144\5{>\2\u0144"+
		"d\3\2\2\2\u0145\u0146\t\t\2\2\u0146f\3\2\2\2\u0147\u0148\t\n\2\2\u0148"+
		"h\3\2\2\2\u0149\u014a\t\13\2\2\u014aj\3\2\2\2\u014b\u014c\5i\65\2\u014c"+
		"\u014d\5i\65\2\u014d\u014e\5i\65\2\u014e\u014f\5i\65\2\u014fl\3\2\2\2"+
		"\u0150\u0151\7^\2\2\u0151\u0152\7w\2\2\u0152\u0153\3\2\2\2\u0153\u015b"+
		"\5k\66\2\u0154\u0155\7^\2\2\u0155\u0156\7W\2\2\u0156\u0157\3\2\2\2\u0157"+
		"\u0158\5k\66\2\u0158\u0159\5k\66\2\u0159\u015b\3\2\2\2\u015a\u0150\3\2"+
		"\2\2\u015a\u0154\3\2\2\2\u015bn\3\2\2\2\u015c\u015e\5s:\2\u015d\u015f"+
		"\5u;\2\u015e\u015d\3\2\2\2\u015e\u015f\3\2\2\2\u015f\u0164\3\2\2\2\u0160"+
		"\u0161\5w<\2\u0161\u0162\5u;\2\u0162\u0164\3\2\2\2\u0163\u015c\3\2\2\2"+
		"\u0163\u0160\3\2\2\2\u0164p\3\2\2\2\u0165\u0166\7\62\2\2\u0166\u0169\t"+
		"\b\2\2\u0167\u016a\5y=\2\u0168\u016a\5{>\2\u0169\u0167\3\2\2\2\u0169\u0168"+
		"\3\2\2\2\u016a\u016b\3\2\2\2\u016b\u016c\5}?\2\u016cr\3\2\2\2\u016d\u016f"+
		"\5w<\2\u016e\u016d\3\2\2\2\u016e\u016f\3\2\2\2\u016f\u0170\3\2\2\2\u0170"+
		"\u0171\7\60\2\2\u0171\u0176\5w<\2\u0172\u0173\5w<\2\u0173\u0174\7\60\2"+
		"\2\u0174\u0176\3\2\2\2\u0175\u016e\3\2\2\2\u0175\u0172\3\2\2\2\u0176t"+
		"\3\2\2\2\u0177\u0179\t\f\2\2\u0178\u017a\t\r\2\2\u0179\u0178\3\2\2\2\u0179"+
		"\u017a\3\2\2\2\u017a\u017b\3\2\2\2\u017b\u017c\5w<\2\u017cv\3\2\2\2\u017d"+
		"\u017f\5[.\2\u017e\u017d\3\2\2\2\u017f\u0180\3\2\2\2\u0180\u017e\3\2\2"+
		"\2\u0180\u0181\3\2\2\2\u0181x\3\2\2\2\u0182\u0184\5{>\2\u0183\u0182\3"+
		"\2\2\2\u0183\u0184\3\2\2\2\u0184\u0185\3\2\2\2\u0185\u0186\7\60\2\2\u0186"+
		"\u018b\5{>\2\u0187\u0188\5{>\2\u0188\u0189\7\60\2\2\u0189\u018b\3\2\2"+
		"\2\u018a\u0183\3\2\2\2\u018a\u0187\3\2\2\2\u018bz\3\2\2\2\u018c\u018e"+
		"\5i\65\2\u018d\u018c\3\2\2\2\u018e\u018f\3\2\2\2\u018f\u018d\3\2\2\2\u018f"+
		"\u0190\3\2\2\2\u0190|\3\2\2\2\u0191\u0193\t\16\2\2\u0192\u0194\t\r\2\2"+
		"\u0193\u0192\3\2\2\2\u0193\u0194\3\2\2\2\u0194\u0195\3\2\2\2\u0195\u0196"+
		"\5w<\2\u0196~\3\2\2\2\u0197\u0198\7^\2\2\u0198\u01a7\t\17\2\2\u0199\u019a"+
		"\7^\2\2\u019a\u019c\5g\64\2\u019b\u019d\5g\64\2\u019c\u019b\3\2\2\2\u019c"+
		"\u019d\3\2\2\2\u019d\u019f\3\2\2\2\u019e\u01a0\5g\64\2\u019f\u019e\3\2"+
		"\2\2\u019f\u01a0\3\2\2\2\u01a0\u01a7\3\2\2\2\u01a1\u01a2\7^\2\2\u01a2"+
		"\u01a3\7z\2\2\u01a3\u01a4\3\2\2\2\u01a4\u01a7\5{>\2\u01a5\u01a7\5m\67"+
		"\2\u01a6\u0197\3\2\2\2\u01a6\u0199\3\2\2\2\u01a6\u01a1\3\2\2\2\u01a6\u01a5"+
		"\3\2\2\2\u01a7\u0080\3\2\2\2\u01a8\u01aa\t\20\2\2\u01a9\u01a8\3\2\2\2"+
		"\u01aa\u01ab\3\2\2\2\u01ab\u01a9\3\2\2\2\u01ab\u01ac\3\2\2\2\u01ac\u01ad"+
		"\3\2\2\2\u01ad\u01ae\bA\2\2\u01ae\u0082\3\2\2\2\u01af\u01b1\7\17\2\2\u01b0"+
		"\u01b2\7\f\2\2\u01b1\u01b0\3\2\2\2\u01b1\u01b2\3\2\2\2\u01b2\u01b5\3\2"+
		"\2\2\u01b3\u01b5\7\f\2\2\u01b4\u01af\3\2\2\2\u01b4\u01b3\3\2\2\2\u01b5"+
		"\u01b6\3\2\2\2\u01b6\u01b7\bB\2\2\u01b7\u0084\3\2\2\2\"\2\u00f9\u00fb"+
		"\u00ff\u0103\u010b\u010f\u0113\u0118\u011d\u0126\u0131\u0137\u013e\u015a"+
		"\u015e\u0163\u0169\u016e\u0175\u0179\u0180\u0183\u018a\u018f\u0193\u019c"+
		"\u019f\u01a6\u01ab\u01b1\u01b4\3\b\2\2";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}