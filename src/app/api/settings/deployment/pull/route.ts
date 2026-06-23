import { execFile } from "node:child_process";
import { mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { promisify } from "node:util";
import type { NextRequest } from "next/server";
import { requireSession, unauthorized } from "@/lib/auth/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const exec = promisify(execFile);

const REPO_URL = process.env.CITIVIA_REPO_URL ?? "https://github.com/riseon-lab/igglepixel.git";
const REPO_REF = process.env.CITIVIA_REPO_REF ?? "main";
const REPO_DIR = process.env.CITIVIA_REPO_DIR ?? "/workspace/igglepixel";

async function run(cmd: string, args: string[], cwd?: string) {
  const { stdout, stderr } = await exec(cmd, args, {
    cwd,
    timeout: 10 * 60_000,
    maxBuffer: 1024 * 1024,
  });
  return [stdout, stderr].filter(Boolean).join("\n").trim();
}

export async function POST(req: NextRequest) {
  if (!(await requireSession(req))) return unauthorized();

  const log: string[] = [];
  try {
    await mkdir(dirname(REPO_DIR), { recursive: true });
    try {
      await run("git", ["-C", REPO_DIR, "rev-parse", "--is-inside-work-tree"]);
      log.push(await run("git", ["-C", REPO_DIR, "remote", "set-url", "origin", REPO_URL]));
      log.push(await run("git", ["-C", REPO_DIR, "pull", "--ff-only", "origin", REPO_REF]));
    } catch {
      log.push(await run("git", ["clone", "--branch", REPO_REF, "--depth", "1", REPO_URL, REPO_DIR]));
    }
    log.push(await run("npm", ["ci", "--include=dev"], REPO_DIR));
    log.push(await run("npm", ["run", "build"], REPO_DIR));
    return Response.json({
      ok: true,
      repoDir: REPO_DIR,
      ref: REPO_REF,
      restartRequired: true,
      log: log.filter(Boolean).join("\n").slice(-4000),
    });
  } catch (err) {
    return Response.json(
      {
        ok: false,
        repoDir: REPO_DIR,
        ref: REPO_REF,
        error: err instanceof Error ? err.message : "Pull failed.",
        log: log.filter(Boolean).join("\n").slice(-4000),
      },
      { status: 500 },
    );
  }
}
