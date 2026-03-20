"use client";

import { useState, useMemo } from "react";
import { nodes, edges, getCoreNodes, getCoreEdges, getChapterNodes, getPaperNodes } from "@/data/graph-data";
import { CATEGORY_COLORS, CATEGORY_LABELS, STATUS_COLORS, EDGE_TYPE_LABELS } from "@/types/graph";
import type { GraphNode, GraphEdge, NodeCategory, EdgeType } from "@/types/graph";
import ForceGraph from "@/components/ForceGraph";

type ViewMode = "core" | "all" | "chapters";

export default function Home() {
  const [viewMode, setViewMode] = useState<ViewMode>("core");
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [hiddenCategories, setHiddenCategories] = useState<Set<NodeCategory>>(new Set());
  const [tab, setTab] = useState<"graph" | "papers" | "chapters">("graph");

  const filteredNodes = useMemo(() => {
    let n: GraphNode[];
    if (viewMode === "core") n = getCoreNodes();
    else if (viewMode === "chapters") n = getChapterNodes();
    else n = nodes;

    if (hiddenCategories.size > 0) {
      n = n.filter((node) => !hiddenCategories.has(node.category));
    }
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      n = n.filter(
        (node) =>
          node.label.toLowerCase().includes(q) ||
          node.description?.toLowerCase().includes(q) ||
          node.id.toLowerCase().includes(q)
      );
    }
    return n;
  }, [viewMode, hiddenCategories, searchQuery]);

  const filteredEdges = useMemo(() => {
    let e: GraphEdge[];
    if (viewMode === "core") e = getCoreEdges();
    else e = edges;

    const nodeIds = new Set(filteredNodes.map((n) => n.id));
    return e.filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target));
  }, [viewMode, filteredNodes]);

  const toggleCategory = (cat: NodeCategory) => {
    setHiddenCategories((prev) => {
      const next = new Set(prev);
      if (next.has(cat)) next.delete(cat);
      else next.add(cat);
      return next;
    });
  };

  const connectedEdges = useMemo(() => {
    if (!selectedNode) return [];
    return edges.filter(
      (e) => e.source === selectedNode.id || e.target === selectedNode.id
    );
  }, [selectedNode]);

  const connectedNodes = useMemo(() => {
    if (!selectedNode) return [];
    const ids = new Set(
      connectedEdges.flatMap((e) => [e.source, e.target])
    );
    ids.delete(selectedNode.id);
    return nodes.filter((n) => ids.has(n.id));
  }, [selectedNode, connectedEdges]);

  const chapterNodes = getChapterNodes().sort(
    (a, b) => (a.chapterNum ?? 0) - (b.chapterNum ?? 0)
  );

  const paperNodes = getPaperNodes().sort(
    (a, b) => (b.year ?? 0) - (a.year ?? 0)
  );

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Left sidebar */}
      <aside className="w-72 flex-shrink-0 border-r border-[var(--border)] flex flex-col overflow-y-auto">
        <div className="p-4 border-b border-[var(--border)]">
          <h1 className="text-lg font-semibold text-[var(--accent)]">
            Dissertation Graph
          </h1>
          <p className="text-xs text-[var(--text-muted)] mt-1">
            Meta-Learning Multi-Agent Policy Gradients
          </p>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-[var(--border)]">
          {(["graph", "chapters", "papers"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`flex-1 px-3 py-2 text-xs font-medium uppercase tracking-wider ${
                tab === t
                  ? "text-[var(--accent)] border-b-2 border-[var(--accent)]"
                  : "text-[var(--text-muted)] hover:text-[var(--text)]"
              }`}
            >
              {t}
            </button>
          ))}
        </div>

        {tab === "graph" && (
          <div className="p-4 space-y-4 flex-1 overflow-y-auto">
            {/* Search */}
            <input
              type="text"
              placeholder="Search nodes..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full px-3 py-2 text-sm bg-[var(--bg-card)] border border-[var(--border)] rounded-lg focus:outline-none focus:border-[var(--accent)]"
            />

            {/* View mode */}
            <div>
              <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-2">
                View
              </p>
              <div className="flex gap-1">
                {(["core", "all", "chapters"] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => setViewMode(m)}
                    className={`flex-1 px-2 py-1.5 text-xs rounded ${
                      viewMode === m
                        ? "bg-[var(--accent)] text-black font-medium"
                        : "bg-[var(--bg-card)] text-[var(--text-muted)] hover:text-[var(--text)]"
                    }`}
                  >
                    {m === "core" ? "Core" : m === "all" ? "All" : "Chapters"}
                  </button>
                ))}
              </div>
            </div>

            {/* Category filters */}
            <div>
              <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-2">
                Categories
              </p>
              <div className="space-y-1">
                {(Object.keys(CATEGORY_LABELS) as NodeCategory[]).map((cat) => {
                  const count = filteredNodes.filter(
                    (n) => n.category === cat
                  ).length;
                  if (count === 0 && hiddenCategories.has(cat)) return null;
                  return (
                    <button
                      key={cat}
                      onClick={() => toggleCategory(cat)}
                      className={`w-full flex items-center gap-2 px-2 py-1.5 text-xs rounded hover:bg-[var(--bg-hover)] ${
                        hiddenCategories.has(cat) ? "opacity-40" : ""
                      }`}
                    >
                      <span
                        className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                        style={{ background: CATEGORY_COLORS[cat] }}
                      />
                      <span className="flex-1 text-left">
                        {CATEGORY_LABELS[cat]}
                      </span>
                      <span className="text-[var(--text-muted)]">{count}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Stats */}
            <div className="text-xs text-[var(--text-muted)] pt-2 border-t border-[var(--border)]">
              {filteredNodes.length} nodes · {filteredEdges.length} edges
            </div>
          </div>
        )}

        {tab === "chapters" && (
          <div className="p-4 space-y-2 flex-1 overflow-y-auto">
            {chapterNodes.map((ch) => (
              <button
                key={ch.id}
                onClick={() => { setSelectedNode(ch); setTab("graph"); }}
                className="w-full flex items-start gap-3 px-3 py-2 rounded-lg hover:bg-[var(--bg-hover)] text-left"
              >
                <span
                  className="mt-1 w-2 h-2 rounded-full flex-shrink-0"
                  style={{
                    background: STATUS_COLORS[ch.status ?? "todo"],
                  }}
                />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{ch.label}</p>
                  <p className="text-xs text-[var(--text-muted)] mt-0.5 line-clamp-2">
                    {ch.description}
                  </p>
                </div>
                <span
                  className="text-[10px] uppercase tracking-wider font-medium mt-1 px-1.5 py-0.5 rounded"
                  style={{
                    background: `${STATUS_COLORS[ch.status ?? "todo"]}22`,
                    color: STATUS_COLORS[ch.status ?? "todo"],
                  }}
                >
                  {ch.status}
                </span>
              </button>
            ))}

            <div className="pt-3 border-t border-[var(--border)]">
              <p className="text-xs text-[var(--text-muted)] mb-2">Progress</p>
              <div className="w-full h-2 bg-[var(--bg-card)] rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${(chapterNodes.filter((c) => c.status === "done" || c.status === "draft").length / chapterNodes.length) * 100}%`,
                    background: `linear-gradient(90deg, ${STATUS_COLORS.done}, ${STATUS_COLORS.draft})`,
                  }}
                />
              </div>
              <p className="text-xs text-[var(--text-muted)] mt-1">
                {chapterNodes.filter((c) => c.status === "done").length} done ·{" "}
                {chapterNodes.filter((c) => c.status === "draft").length} draft ·{" "}
                {chapterNodes.filter((c) => c.status === "todo" || c.status === "outline").length} remaining
              </p>
            </div>
          </div>
        )}

        {tab === "papers" && (
          <div className="p-4 space-y-2 flex-1 overflow-y-auto">
            {paperNodes.map((p) => (
              <button
                key={p.id}
                onClick={() => { setSelectedNode(p); setTab("graph"); }}
                className="w-full text-left px-3 py-2 rounded-lg hover:bg-[var(--bg-hover)]"
              >
                <p className="text-sm font-medium">{p.label}</p>
                {p.authors && (
                  <p className="text-xs text-[var(--text-muted)] mt-0.5 truncate">
                    {p.authors.slice(0, 3).join(", ")}
                    {p.authors.length > 3 ? " et al." : ""}
                  </p>
                )}
                {p.venue && (
                  <p className="text-xs text-[var(--text-muted)]">
                    {p.venue} {p.year && `(${p.year})`}
                  </p>
                )}
                <div className="flex gap-1 mt-1 flex-wrap">
                  {p.chapters?.map((ch) => (
                    <span
                      key={ch}
                      className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--accent-dim)] text-[var(--accent)]"
                    >
                      Ch.{ch.match(/ch(\d+)/)?.[1]}
                    </span>
                  ))}
                </div>
              </button>
            ))}
          </div>
        )}
      </aside>

      {/* Main content: graph */}
      <main className="flex-1 relative">
        <ForceGraph
          nodes={filteredNodes}
          edges={filteredEdges}
          selectedNode={selectedNode}
          onSelectNode={setSelectedNode}
        />

        {/* Selected node detail panel */}
        {selectedNode && (
          <div className="absolute top-4 right-4 w-80 bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5 shadow-2xl overflow-y-auto max-h-[calc(100vh-32px)]">
            <div className="flex items-start justify-between mb-3">
              <div>
                <span
                  className="text-[10px] uppercase tracking-wider font-medium px-2 py-0.5 rounded-full"
                  style={{
                    background: `${CATEGORY_COLORS[selectedNode.category]}22`,
                    color: CATEGORY_COLORS[selectedNode.category],
                  }}
                >
                  {CATEGORY_LABELS[selectedNode.category]}
                </span>
                <h2 className="text-lg font-semibold mt-2">
                  {selectedNode.label}
                </h2>
              </div>
              <button
                onClick={() => setSelectedNode(null)}
                className="text-[var(--text-muted)] hover:text-[var(--text)] text-lg leading-none"
              >
                ×
              </button>
            </div>

            {selectedNode.description && (
              <p className="text-sm text-[var(--text-muted)] mb-4">
                {selectedNode.description}
              </p>
            )}

            {selectedNode.formalStatement && (
              <div className="mb-4 px-3 py-2 bg-[var(--bg)] rounded-lg border border-[var(--border)]">
                <p className="text-xs text-[var(--text-muted)] mb-1 uppercase tracking-wider">
                  Formal statement
                </p>
                <code className="text-xs text-[var(--accent)] break-all">
                  {selectedNode.formalStatement}
                </code>
              </div>
            )}

            {selectedNode.status && (
              <div className="mb-4">
                <span
                  className="text-xs font-medium px-2 py-1 rounded"
                  style={{
                    background: `${STATUS_COLORS[selectedNode.status]}22`,
                    color: STATUS_COLORS[selectedNode.status],
                  }}
                >
                  {selectedNode.status.toUpperCase()}
                </span>
              </div>
            )}

            {selectedNode.authors && (
              <div className="mb-4">
                <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">
                  Authors
                </p>
                <p className="text-sm">{selectedNode.authors.join(", ")}</p>
              </div>
            )}

            {selectedNode.url && (
              <a
                href={selectedNode.url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-block text-xs text-[var(--accent)] hover:underline mb-4"
              >
                Open paper →
              </a>
            )}

            {/* Connections */}
            {connectedNodes.length > 0 && (
              <div>
                <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-2">
                  Connections ({connectedEdges.length})
                </p>
                <div className="space-y-1.5">
                  {connectedEdges.map((e, i) => {
                    const otherId =
                      e.source === selectedNode.id ? e.target : e.source;
                    const other = nodes.find((n) => n.id === otherId);
                    if (!other) return null;
                    const direction =
                      e.source === selectedNode.id ? "→" : "←";
                    return (
                      <button
                        key={i}
                        onClick={() => setSelectedNode(other)}
                        className="w-full flex items-center gap-2 px-2 py-1.5 text-xs rounded hover:bg-[var(--bg-hover)] text-left"
                      >
                        <span
                          className="w-2 h-2 rounded-full flex-shrink-0"
                          style={{
                            background: CATEGORY_COLORS[other.category],
                          }}
                        />
                        <span className="text-[var(--text-muted)]">
                          {direction}
                        </span>
                        <span className="flex-1 truncate">{other.label}</span>
                        <span className="text-[var(--text-muted)]">
                          {EDGE_TYPE_LABELS[e.type]}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
