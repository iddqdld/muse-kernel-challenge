// Append one papercut line to PAPERCUTS.md.
// Usage: yarn papercut -m <model> <message>

import { appendFileSync } from "node:fs";
import { resolve } from "node:path";

const args = process.argv.slice(2);
const modelFlag = args.indexOf("-m");

if (modelFlag === -1 || !args[modelFlag + 1]) {
  console.error("Use: yarn papercut -m <model> <message>");
  process.exit(2);
}

const model = args[modelFlag + 1];
const message = args
  .filter((_, index) => index !== modelFlag && index !== modelFlag + 1)
  .join(" ")
  .replace(/\s+/g, " ")
  .trim();

if (!message) {
  console.error("The papercut message is empty.");
  process.exit(2);
}

const timestamp = new Date().toISOString();
appendFileSync(resolve("PAPERCUTS.md"), `\n- ${timestamp} [${model}] ${message}\n`, "utf8");
