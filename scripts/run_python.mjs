import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";

const python = process.platform === "win32" ? ".venv\\Scripts\\python.exe" : ".venv/bin/python";
if (!existsSync(python)) {
  console.error(`未找到 ${python}。请先运行 handoff 目录中的系统启动脚本，或手动创建虚拟环境。`);
  process.exit(1);
}

const result = spawnSync(python, process.argv.slice(2), { stdio: "inherit" });
if (result.error) {
  console.error(result.error.message);
  process.exit(1);
}
process.exit(result.status ?? 1);
