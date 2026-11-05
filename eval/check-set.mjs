// Validates an eval set without touching the model: every hand-written reference regex must pass its
// own strings, so a failure here is a mistake in the test, not the model.
import { readFileSync } from "node:fs";

const file = process.argv[2];
const cases = JSON.parse(readFileSync(file));
let bad = 0, weak = 0;
const seen = new Set();
for (const c of cases) {
  if (seen.has(c.q)) { console.log("DUPLICATE", c.q); bad++; }
  seen.add(c.q);
  if (!c.no.length) weak++;
  if (!c.ref) continue;
  const re = new RegExp(c.ref, (c.flags ?? "") + "m");
  const wrongYes = c.yes.filter((s) => !re.test(s)), wrongNo = c.no.filter((s) => re.test(s));
  if (wrongYes.length || wrongNo.length) {
    bad++;
    console.log(`BAD  ${c.q}  /${c.ref}/  should match: ${JSON.stringify(wrongYes)}  shouldn't: ${JSON.stringify(wrongNo)}`);
  }
}
const cats = {};
for (const c of cases) cats[c.c ?? "-"] = (cats[c.c ?? "-"] ?? 0) + 1;
console.log(`${file}: ${cases.length} prompts, ${bad} bad, ${weak} with no negative strings`, cats);
process.exit(bad ? 1 : 0);
