"""Phrase tables shared by the data generator and (via export) the JS compiler.

The model only tags a span as ATOM; this table is what decides that "lower case letter" means [a-z].
"""
import json
import re

# Macro tokens: one output token that expands to a whole regex.
PRESETS = {
    "<EMAIL>": (r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
                ["email", "email address", "e-mail", "e-mail address", "mail address", "emails", "email addresses"]),
    "<URL>": (r"https?://[^\s/$.?#][^\s]*",
              ["url", "urls", "link", "links", "web address", "website address", "http url", "http or https url", "web link"]),
    "<IPV4>": (r"(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)",
               ["ip address", "ip addresses", "ipv4 address", "ipv4", "ip"]),
    "<IPV6>": (r"(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}", ["ipv6 address", "ipv6"]),
    "<DATE_ISO>": (r"\d{4}-\d{2}-\d{2}",
                   ["date", "dates", "iso date", "yyyy-mm-dd date", "date in iso format", "date like yyyy-mm-dd", "yyyy-mm-dd"]),
    "<DATE_US>": (r"\d{1,2}/\d{1,2}/\d{4}", ["mm/dd/yyyy date", "us date", "american date", "date like mm/dd/yyyy", "mm/dd/yyyy", "dd/mm/yyyy", "dd/mm/yyyy date"]),
    "<TIME>": (r"(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?", ["time", "times", "24 hour time", "time of day", "hh:mm time", "hh:mm", "timestamp"]),
    "<PHONE>": (r"\+?\d{0,2}[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}",
                ["phone number", "phone numbers", "telephone number", "us phone number", "mobile number", "phone"]),
    "<HEXCOLOR>": (r"#(?:[0-9a-fA-F]{3}){1,2}\b", ["hex color", "hex colors", "css color", "hex colour", "html color", "color code", "hex color code"]),
    "<UUID>": (r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", ["uuid", "uuids", "guid", "guids"]),
    "<ZIP>": (r"\d{5}(?:-\d{4})?", ["zip code", "zip codes", "zip", "us zip code", "postal code", "postcode"]),
    "<CARD>": (r"\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}", ["credit card number", "credit card", "card number", "credit card numbers"]),
    "<SSN>": (r"\d{3}-\d{2}-\d{4}", ["ssn", "social security number"]),
    "<MAC>": (r"(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}", ["mac address", "mac addresses"]),
    "<SEMVER>": (r"\d+\.\d+\.\d+", ["semver", "semantic version", "version number", "version"]),
    "<SLUG>": (r"[a-z0-9]+(?:-[a-z0-9]+)*", ["slug", "url slug", "slugs"]),
    "<HTMLTAG>": (r"<[^>]+>", ["html tag", "html tags", "xml tag", "xml tags", "tag"]),
    "<INT>": (r"-?\d+", ["integer", "integers", "whole number", "signed integer", "negative or positive integer"]),
    "<FLOAT>": (r"-?\d+\.\d+", ["float", "floats", "decimal", "decimals", "decimal number", "decimal numbers", "floating point number"]),
    "<PERCENT>": (r"\d+(?:\.\d+)?%", ["percentage", "percentages", "percent", "percent value"]),
    "<MONEY>": (r"\$\d+(?:\.\d{2})?", ["price", "prices", "dollar amount", "dollar amounts", "money amount", "currency", "price in dollars"]),
    "<HASHTAG>": (r"#\w+", ["hashtag", "hashtags"]),
    "<MENTION>": (r"@\w+", ["mention", "mentions", "at mention", "twitter handle", "twitter handles"]),
    "<USERNAME>": (r"[a-zA-Z][a-zA-Z0-9_-]{2,15}", ["username", "usernames", "user name", "login name"]),
    "<DQUOTED>": (r"\"[^\"]*\"", ["quoted string", "quoted strings", "double quoted string", "text in double quotes", "text in quotes", "string in quotes"]),
    "<PARENS>": (r"\([^)]*\)", ["text in parentheses", "parenthesized text", "something in parentheses", "text inside parentheses"]),
    "<BRACKETS>": (r"\[[^\]]*\]", ["text in brackets", "text in square brackets", "text inside brackets"]),
    "<DUPWORD>": (r"\b(\w+)\s+\1\b", ["duplicate words", "repeated words", "doubled words", "same word twice", "duplicate word", "word repeated twice in a row"]),
    "<TRAILWS>": (r"[ \t]+$", ["trailing whitespace", "trailing spaces", "whitespace at the end of a line", "spaces at the end of the line"]),
    "<LEADWS>": (r"^[ \t]+", ["leading whitespace", "leading spaces", "indentation", "whitespace at the start of a line"]),
    "<EMPTYLINE>": (r"^$", ["empty line", "empty lines", "empty string"]),
    "<BLANKLINE>": (r"^\s*$", ["blank line", "blank lines", "line with only whitespace", "whitespace only line"]),
    "<BASE64>": (r"(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?", ["base64", "base64 string", "base64 encoded string"]),
    "<JWT>": (r"[\w-]+\.[\w-]+\.[\w-]+", ["jwt", "jwt token", "json web token"]),
    "<DOMAIN>": (r"(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}", ["domain", "domain name", "domains", "hostname", "domain names"]),
    "<FILEPATH>": (r"(?:/[^/\s]+)+/?", ["unix path", "file path", "filepath", "path"]),
    "<CAMEL>": (r"\b[a-z]+(?:[A-Z][a-z0-9]*)+\b", ["camelcase word", "camel case word", "camelcase", "camel case identifier"]),
    "<SNAKE>": (r"\b[a-z]+(?:_[a-z0-9]+)+\b", ["snake case word", "snake_case", "snake case identifier", "snake case"]),
    "<IDENT>": (r"\b[a-zA-Z_]\w*\b", ["identifier", "identifiers", "variable name", "variable names"]),
    "<COMMENT>": (r"//.*$", ["line comment", "javascript comment", "c style comment", "code comment", "comment"]),
    "<CAPWORD>": (r"\b[A-Z][a-z]+\b", ["capitalized word", "capitalized words", "word starting with a capital letter", "title case word", "proper noun"]),
    "<WORD>": (r"\b\w+\b", ["word", "words", "any word", "whole word"]),
    "<ALLCAPS>": (r"\b[A-Z]{2,}\b", ["all caps word", "all caps words", "uppercase word", "uppercase words", "acronym", "acronyms", "shouting"]),
}

# (tokens, singular phrases, plural phrases, inner token usable inside [...])
CLASSES = [
    (r"\d", ["digit", "number character", "numeric character", "numeral"], ["digits", "numerals", "numeric characters"], r"\d"),
    ("[a-zA-Z]", ["letter", "alphabetic character", "alpha character"], ["letters", "alphabetic characters", "alpha characters"], "a-zA-Z"),
    ("[a-z]", ["lowercase letter", "lower case letter", "small letter", "lowercase character"], ["lowercase letters", "lower case letters", "small letters", "lowercase characters", "lowercase"], "a-z"),
    ("[A-Z]", ["uppercase letter", "upper case letter", "capital letter", "capital", "uppercase character"], ["uppercase letters", "upper case letters", "capital letters", "capitals", "uppercase characters", "uppercase"], "A-Z"),
    ("[a-zA-Z0-9]", ["alphanumeric character", "alphanumeric", "letter or digit", "letter or number"], ["alphanumeric characters", "alphanumerics", "letters and digits", "letters and numbers", "letters or digits", "digits and letters"], "a-zA-Z0-9"),
    (r"\w", ["word character", "word char"], ["word characters", "word chars"], r"\w"),
    (r"\s", ["whitespace", "whitespace character", "space or tab"], ["whitespace characters", "whitespace", "spaces or tabs"], r"\s"),
    (r"\D", ["non-digit", "non digit", "non-numeric character"], ["non-digits", "non digits", "non-numeric characters"], r"\D"),
    (r"\W", ["non-word character", "non word character"], ["non-word characters", "non word characters"], r"\W"),
    (r"\S", ["non-whitespace character", "non whitespace character", "non-space character", "visible character"], ["non-whitespace characters", "non whitespace characters", "non-space characters", "visible characters"], r"\S"),
    (".", ["character", "any character", "char", "any char", "anything"], ["characters", "any characters", "chars", "anything"], None),
    ("[0-9a-fA-F]", ["hex digit", "hexadecimal digit", "hex character"], ["hex digits", "hexadecimal digits", "hex characters", "hex"], "0-9a-fA-F"),
    ("[aeiouAEIOU]", ["vowel"], ["vowels"], "aeiouAEIOU"),
    ("[b-df-hj-np-tv-zB-DF-HJ-NP-TV-Z]", ["consonant"], ["consonants"], "b-df-hj-np-tv-zB-DF-HJ-NP-TV-Z"),
    (r"[^\w\s]", ["punctuation mark", "punctuation character", "special character", "symbol"], ["punctuation", "punctuation marks", "punctuation characters", "special characters", "symbols"], None),
]

# Named punctuation. Tokens are already regex-escaped.
PUNCT = [
    (r"\.", ["dot", "period", "full stop", "point"], ["dots", "periods"]),
    (",", ["comma"], ["commas"]),
    ("-", ["dash", "hyphen", "minus sign", "minus"], ["dashes", "hyphens"]),
    ("_", ["underscore"], ["underscores"]),
    (":", ["colon"], ["colons"]),
    (";", ["semicolon"], ["semicolons"]),
    ("/", ["slash", "forward slash"], ["slashes", "forward slashes"]),
    (r"\\", ["backslash"], ["backslashes"]),
    ("@", ["at sign", "at symbol"], ["at signs"]),
    ("#", ["hash", "hash sign", "pound sign", "hashtag symbol"], ["hashes"]),
    (r"\$", ["dollar sign", "dollar"], ["dollar signs"]),
    ("%", ["percent sign"], ["percent signs"]),
    (r"\+", ["plus sign", "plus"], ["plus signs"]),
    (r"\*", ["asterisk", "star"], ["asterisks", "stars"]),
    (r"\?", ["question mark"], ["question marks"]),
    ("=", ["equals sign", "equal sign", "equals"], ["equals signs"]),
    (" ", ["space", "space character", "single space"], ["spaces"]),
    (r"\t", ["tab", "tab character"], ["tabs"]),
    ('"', ["double quote", "quote", "quotation mark"], ["double quotes", "quotes", "quotation marks"]),
    ("'", ["single quote", "apostrophe"], ["single quotes", "apostrophes"]),
    (r"\(", ["open paren", "opening parenthesis", "left parenthesis"], ["opening parentheses"]),
    (r"\)", ["close paren", "closing parenthesis", "right parenthesis"], ["closing parentheses"]),
    (r"\[", ["opening bracket", "left bracket"], ["opening brackets"]),
    (r"\]", ["closing bracket", "right bracket"], ["closing brackets"]),
    (r"\|", ["pipe", "vertical bar"], ["pipes"]),
    ("!", ["exclamation mark"], ["exclamation marks"]),
    ("&", ["ampersand"], ["ampersands"]),
]

COMMON_WORDS = """foo bar baz cat dog bird fish hello world error warning info debug todo fixme test
user admin root server client data file image photo video http https ftp www api json xml html css
apple banana orange red green blue black white yes no true false null none class def function return
import export const var let begin stop start finish north south east west monday friday january march
ing ed tion ly un pre re er est ness ment able ful less ion anti com org net edu gov png jpg gif pdf txt
csv mp3 mp4 zip tar log py js ts rs go java abc xyz qwerty lorem ipsum spam eggs alice bob carol dave
TODO FIXME ERROR WARN INFO DEBUG GET POST PUT DELETE USD EUR""".split()
SYMBOL_LITS = [".png", ".jpg", ".txt", ".js", ".com", ".org", "foo.bar", "foo-bar", "foo_bar", "a+b", "c++", "$HOME", "#include", "@home", "->", "=>", "::", "!=", "==", "..", "//", "http://", "https://", "[x]", "(a)", "Mr.", "v1", "x86", "h1"]
SYLL = "ba be bi bo da de di do fa fe fi ga go ka ke ki ko la le li lo ma me mi mo na ne ni no pa pe pi po ra re ri ro sa se si so ta te ti to va ve vi vo za ze zi zo qu tr st pl ch sh th".split()

PRESETS["<NUMBER>"] = (r"\d+", ["number", "numbers", "positive number", "positive integer", "positive integers", "natural number", "unsigned integer"])
PRESETS["<KEYVALUE>"] = (r"\w+=\S+", ["key value pair", "key value pairs", "key=value", "key equals value", "key equals value pair"])


def compact(form):
    low = form.lower()
    return re.sub(r"[^a-z0-9]", "", low) or low


def build_lexicon():
    """compact form -> {src, inner?, plural?, preset?}. A form used as both singular and plural is singular."""
    lex = {}
    for src, sg, pl in PUNCT:
        inner = {"-": r"\-"}.get(src, src)
        for f in pl:
            lex[compact(f)] = dict(src=src, inner=inner, plural=True)
        for f in sg:
            lex[compact(f)] = dict(src=src, inner=inner, plural=False)
    for src, sg, pl, inner in CLASSES:
        for f in pl:
            lex[compact(f)] = dict(src=src, inner=inner, plural=True)
        for f in sg:
            lex[compact(f)] = dict(src=src, inner=inner, plural=False)
    for _, (src, forms) in PRESETS.items():
        for f in forms:
            lex[compact(f)] = dict(src=src, preset=True)
    return lex


LEX = build_lexicon()

if __name__ == "__main__":
    print(len(LEX), "lexicon forms")
    print(json.dumps(LEX)[:200])
