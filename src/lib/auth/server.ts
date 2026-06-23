import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import {
  createAccount,
  getAccount,
  login,
  logout,
  tokensMatch,
  type AccountRecord,
} from "./store";

export const SESSION_COOKIE = "citivia_session";
const SESSION_MAX_AGE = 60 * 60 * 24 * 30; // 30 days
const secureCookies = () => process.env.CITIVIA_SECURE_COOKIES === "1";

/** Returns the account if the request carries the current valid session, else null. */
export async function requireSession(
  req: NextRequest,
): Promise<AccountRecord | null> {
  const token = req.cookies.get(SESSION_COOKIE)?.value;
  const acc = await getAccount();
  if (!token || !acc?.sessionToken) return null;
  return tokensMatch(token, acc.sessionToken) ? acc : null;
}

export function setSessionCookie(res: NextResponse, token: string): void {
  res.cookies.set(SESSION_COOKIE, token, {
    httpOnly: true,
    sameSite: "lax",
    secure: secureCookies(),
    path: "/",
    maxAge: SESSION_MAX_AGE,
  });
}

export function clearSessionCookie(res: NextResponse): void {
  res.cookies.set(SESSION_COOKIE, "", {
    httpOnly: true,
    sameSite: "lax",
    secure: secureCookies(),
    path: "/",
    maxAge: 0,
  });
}

/** Shared 401 for guarded API routes. */
export function unauthorized(): Response {
  return new Response("Unauthorized", { status: 401 });
}

export { createAccount, getAccount, login, logout };
export type { AccountRecord };
