"use client";

// In-memory, per-model workspace state (queue + reference images) that survives
// client-side navigation. The generation workspace component unmounts whenever
// you leave its route, which used to wipe the queue; holding the state up here in
// a provider mounted on the app layout keeps it alive until a full page reload.

import {
  createContext,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { QueueJob } from "@/lib/types";

export interface WorkspaceRef {
  name: string;
  dataUrl: string;
}

interface ModelWorkspace {
  jobs: QueueJob[];
  references: WorkspaceRef[];
  refIndex: number;
}

const EMPTY: ModelWorkspace = { jobs: [], references: [], refIndex: 0 };

interface StoreValue {
  state: Record<string, ModelWorkspace>;
  setJobs: (model: string, update: (jobs: QueueJob[]) => QueueJob[]) => void;
  setReferences: (
    model: string,
    update: (refs: WorkspaceRef[]) => WorkspaceRef[],
  ) => void;
  setRefIndex: (model: string, index: number) => void;
}

const Ctx = createContext<StoreValue | null>(null);

export function WorkspaceStoreProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<Record<string, ModelWorkspace>>({});

  const value = useMemo<StoreValue>(
    () => ({
      state,
      setJobs: (model, update) =>
        setState((prev) => {
          const cur = prev[model] ?? EMPTY;
          return { ...prev, [model]: { ...cur, jobs: update(cur.jobs) } };
        }),
      setReferences: (model, update) =>
        setState((prev) => {
          const cur = prev[model] ?? EMPTY;
          const references = update(cur.references);
          const refIndex = Math.min(
            cur.refIndex,
            Math.max(0, references.length - 1),
          );
          return { ...prev, [model]: { ...cur, references, refIndex } };
        }),
      setRefIndex: (model, index) =>
        setState((prev) => {
          const cur = prev[model] ?? EMPTY;
          return { ...prev, [model]: { ...cur, refIndex: index } };
        }),
    }),
    [state],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

/** Bound slice of the store for a single model. */
export function useModelWorkspace(model: string) {
  const store = useContext(Ctx);
  if (!store) {
    throw new Error("useModelWorkspace must be used within WorkspaceStoreProvider");
  }
  const ws = store.state[model] ?? EMPTY;
  return {
    jobs: ws.jobs,
    references: ws.references,
    refIndex: ws.refIndex,
    setJobs: (update: (jobs: QueueJob[]) => QueueJob[]) =>
      store.setJobs(model, update),
    setReferences: (update: (refs: WorkspaceRef[]) => WorkspaceRef[]) =>
      store.setReferences(model, update),
    setRefIndex: (index: number) => store.setRefIndex(model, index),
  };
}
