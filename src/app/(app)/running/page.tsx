import { redirect } from "next/navigation";

// Consolidated into /models. Kept as a redirect for old links/bookmarks.
export default function RunningRedirect() {
  redirect("/models");
}
