"""Text that isn't a request for a pattern. Every token is labelled O, so the compiler returns no regex.

A lone word stays a valid literal ("hello"), so negatives have at least three tokens or are symbols.
"""
SALAD = """time year people way day man woman child world life hand part place week case point company number
group problem fact house room school water car city night money story month book eye job friend family
food door river tree garden kitchen table chair window street road bridge mountain forest ocean island farm
village market store shop hospital church museum library station airport hotel restaurant office factory
doctor teacher student driver farmer artist singer writer lawyer nurse pilot soldier
horse cow sheep chicken rabbit lion tiger elephant monkey bear wolf duck
apple orange banana bread cheese butter rice soup salad cake tea juice milk wine beer sugar
shirt shoe hat coat dress jacket glove sock
guitar piano drum song movie picture photo camera phone computer radio television
summer winter spring autumn morning evening afternoon yesterday tomorrow today tonight monday sunday
walk run jump swim sing dance read write draw paint cook eat drink sleep drive fly ride climb
buy sell pay give take bring carry send receive open close wash clean fix build break cut
think know believe hope want love hate like enjoy remember forget understand learn teach
said went came saw made took gave found told asked worked called tried needed felt became left kept began
big small tall short long old young new good bad happy sad angry tired hungry cold hot warm cool
fast slow easy hard early late cheap expensive beautiful ugly quiet loud bright dark clean dirty
red blue green yellow purple brown black white pink gray
he she they we you i his her their our your him them us
very quite really always never often sometimes usually maybe perhaps again soon still already
because although while before after during until since though
""".split()
FILLER = "the a an of in on at to for with from by and or but is are was were be been has have had do does did will would can could should".split()
STARTS = {
    "what": ["what is", "what are", "what did", "what does", "what would"],
    "how": ["how is", "how are", "how many", "how long", "how did", "how does"],
    "who": ["who is", "who was", "who did"],
    "where": ["where is", "where are", "where did"],
    "when": ["when is", "when did", "when will"],
    "why": ["why is", "why are", "why did"],
    "yes": ["do you", "did you", "can you", "could you", "would you", "will you", "should i", "is it", "are you"],
}
CHAT = """hi there | hey there everyone | hello my dear friend | good evening to you | good afternoon all |
thanks so much | thank you for everything | many thanks to you | you are welcome | no problem at all |
how have you been | what is going on | long time no see | take care of yourself | see you next week |
have a nice day | enjoy your meal | good luck with that | i am so sorry | excuse me please |
congratulations on the wedding | merry christmas to all | i love this | that sounds great | sounds good to me |
tell me about yourself | write me a poem | translate this for me | summarize this article | what can you do""".replace("\n", " ").split(" | ")
CODE_NAMES = "foo bar baz data item value result user name count total index node list args self main app config".split()
COMMANDS = ["npm install", "pip install", "git push", "git pull origin", "cargo build", "docker run", "brew install", "make clean", "cd src", "ls -la", "rm -rf", "chmod +x", "kubectl get"]
FOREIGN = """le la les un une des et est dans pour avec sur nous vous ils elle bonsoir merci beaucoup maison chat chien pain vin
der die das ein eine und ist nicht mit auf ich du wir sie danke sehr haus hund katze brot morgen abend
el los las una por para con muy gracias casa perro gato pan agua noche buenos dias tengo quiero
il lo gli sono della grazie mille ciao casa cane gatto pane vino sera buona""".split()
CYRILLIC = "да нет это как что где кто дом кот собака хлеб вода утро вечер спасибо пожалуйста хорошо очень".split()
CJK = "我你他的是不了在有人这中大为上个国们以要时来用生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经"
HANGUL_KANA = "안녕하세요 감사합니다 사랑해요 こんにちは ありがとう さようなら おはよう".split()
KEY_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm", "poiuytrewq", "lkjhgfdsa", "mnbvcxz"]
SYMBOLS = list("!?*%^&~|/\\-+=@#$.,:;")
EMOJI = ["😀", "😂", "🙂", "😍", "👍", "🔥", "🎉", "❤️", "🤔", "😢"]


class Negatives:
    def __init__(self, rng, syllables, is_grammar_word):
        self.r = rng
        self.syllables = syllables
        # Words the grammar uses with a real role would get contradictory labels, as in experiment E1.
        self.salad = [w for w in SALAD if not is_grammar_word(w)]

    def c(self, items):
        return self.r.choice(items)

    def mash(self):
        r, n = self.r, self.r.randint(3, 6)
        kind = r.random()
        if kind < 0.35:
            def word():
                row = self.c(KEY_ROWS)
                k = r.randint(3, 7)
                i = r.randrange(len(row) - k + 1) if k <= len(row) else 0
                return row[i:i + k]
        elif kind < 0.55:
            def word():
                return self.c("abcdefghijklmnopqrstuvwxyz") * r.randint(3, 6)
        elif kind < 0.8:
            def word():
                return "".join(self.c(self.syllables) for _ in range(r.randint(2, 4)))
        else:
            def word():
                return "".join(self.c("bcdfghjklmnpqrstvwxz") if r.random() < 0.8 else self.c("aeiou") for _ in range(r.randint(3, 8)))
        return " ".join(word() for _ in range(n))

    def sentence(self):
        words = []
        for _ in range(self.r.randint(3, 11)):
            words.append(self.c(FILLER) if self.r.random() < 0.3 else self.c(self.salad))
        return " ".join(words)

    def question(self):
        stem = self.c(self.c(list(STARTS.values())))
        return stem + " " + self.sentence_tail()

    def sentence_tail(self):
        return " ".join(self.c(FILLER) if self.r.random() < 0.25 else self.c(self.salad) for _ in range(self.r.randint(2, 7)))

    def chat(self):
        return self.c(CHAT)

    def code(self):
        n = lambda: self.c(CODE_NAMES)
        forms = [
            lambda: f"{n()} = {n()}({n()})", lambda: f"def {n()}({n()}, {n()}): return {n()}", lambda: f"print({n()})",
            lambda: f"if {n()} == {self.r.randint(0, 9)}: {n()} = {self.r.randint(0, 9)}", lambda: f"for {n()} in {n()}: print({n()})",
            lambda: f"const {n()} = {n()}.{n()}({n()});", lambda: f"{self.c(COMMANDS)} {n()}", lambda: f"SELECT {n()} FROM {n()} WHERE {n()} = {self.r.randint(0, 99)}",
            lambda: f"<{self.c(['div', 'span', 'p'])} class=\"{n()}\">{n()}</{self.c(['div', 'span', 'p'])}>", lambda: f"while ({n()} < {self.r.randint(1, 99)}) {n()}++",
            lambda: f"import {n()} from '{n()}'", lambda: f"{n()}.{n()}({n()}).{n()}()",
        ]
        return self.c(forms)()

    def foreign(self):
        k = self.r.random()
        if k < 0.5:
            return " ".join(self.c(FOREIGN) for _ in range(self.r.randint(3, 7)))
        if k < 0.75:
            return " ".join(self.c(CYRILLIC) for _ in range(self.r.randint(2, 6)))
        if k < 0.9:
            return "".join(self.c(CJK) for _ in range(self.r.randint(4, 12)))
        return " ".join(self.c(HANGUL_KANA) for _ in range(self.r.randint(1, 3)))

    def symbols(self):
        if self.r.random() < 0.25:
            return " ".join(self.c(EMOJI) for _ in range(self.r.randint(2, 5)))
        return " ".join(self.c(SYMBOLS) * self.r.randint(2, 4) for _ in range(self.r.randint(2, 4)))

    def sample(self):
        kinds = [(self.mash, 22), (self.sentence, 26), (self.question, 18), (self.chat, 6), (self.code, 12), (self.foreign, 8), (self.symbols, 8)]
        fns, weights = zip(*kinds)
        return self.r.choices(fns, weights)[0]()
