"use client";

import { useRef, useEffect, useCallback, useState } from "react";
import {
  forceSimulation,
  forceLink,
  forceManyBody,
  forceCenter,
  forceCollide,
  forceX,
  forceY,
  type SimulationNodeDatum,
  type SimulationLinkDatum,
} from "d3-force";
import { CATEGORY_COLORS, EDGE_TYPE_COLORS } from "@/types/graph";
import type { GraphNode, GraphEdge } from "@/types/graph";

interface Props {
  nodes: GraphNode[];
  edges: GraphEdge[];
  selectedNode: GraphNode | null;
  onSelectNode: (node: GraphNode | null) => void;
}

type SimNode = GraphNode & SimulationNodeDatum;
type SimLink = SimulationLinkDatum<SimNode> & Omit<GraphEdge, "source" | "target">;

export default function ForceGraph({ nodes, edges, selectedNode, onSelectNode }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const simRef = useRef<ReturnType<typeof forceSimulation<SimNode>> | null>(null);
  const [simNodes, setSimNodes] = useState<SimNode[]>([]);
  const [simLinks, setSimLinks] = useState<SimLink[]>([]);
  const [hoveredNode, setHoveredNode] = useState<SimNode | null>(null);
  const [transform, setTransform] = useState({ x: 0, y: 0, k: 1 });
  const [dragging, setDragging] = useState<SimNode | null>(null);
  const [panning, setPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [dimensions, setDimensions] = useState({ w: 800, h: 600 });

  // Initialise simulation
  useEffect(() => {
    const simN: SimNode[] = nodes.map((n) => ({ ...n }));
    const nodeMap = new Map(simN.map((n) => [n.id, n]));
    const simL: SimLink[] = edges
      .filter((e) => nodeMap.has(e.source) && nodeMap.has(e.target))
      .map((e) => ({
        ...e,
        source: nodeMap.get(e.source as string)!,
        target: nodeMap.get(e.target as string)!,
      }));

    const sim = forceSimulation<SimNode>(simN)
      .force(
        "link",
        forceLink<SimNode, SimLink>(simL)
          .id((d) => d.id)
          .distance(80)
          .strength((d) => (d as SimLink).strength ?? 0.3)
      )
      .force("charge", forceManyBody().strength(-200).distanceMax(400))
      .force("center", forceCenter(dimensions.w / 2, dimensions.h / 2))
      .force("collide", forceCollide<SimNode>().radius(20))
      .force("x", forceX(dimensions.w / 2).strength(0.03))
      .force("y", forceY(dimensions.h / 2).strength(0.03))
      .alphaDecay(0.02)
      .on("tick", () => {
        setSimNodes([...simN]);
        setSimLinks([...simL]);
      });

    simRef.current = sim;

    return () => {
      sim.stop();
    };
  }, [nodes, edges, dimensions.w, dimensions.h]);

  // Resize
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const obs = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        setDimensions({ w: width, h: height });
      }
    });
    obs.observe(canvas.parentElement!);
    return () => obs.disconnect();
  }, []);

  // Draw
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = dimensions.w * dpr;
    canvas.height = dimensions.h * dpr;
    canvas.style.width = `${dimensions.w}px`;
    canvas.style.height = `${dimensions.h}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.clearRect(0, 0, dimensions.w, dimensions.h);
    ctx.save();
    ctx.translate(transform.x, transform.y);
    ctx.scale(transform.k, transform.k);

    const selectedId = selectedNode?.id;
    const hoveredId = hoveredNode?.id;
    const highlightIds = new Set<string>();
    if (selectedId) {
      highlightIds.add(selectedId);
      edges.forEach((e) => {
        const src = typeof e.source === "string" ? e.source : (e.source as SimNode).id;
        const tgt = typeof e.target === "string" ? e.target : (e.target as SimNode).id;
        if (src === selectedId) highlightIds.add(tgt);
        if (tgt === selectedId) highlightIds.add(src);
      });
    }

    // Draw edges
    for (const link of simLinks) {
      const s = link.source as SimNode;
      const t = link.target as SimNode;
      if (s.x == null || t.x == null) continue;

      const isHighlighted =
        selectedId &&
        (highlightIds.has(s.id) && highlightIds.has(t.id));
      const dim = selectedId && !isHighlighted;

      ctx.beginPath();
      ctx.moveTo(s.x, s.y!);
      ctx.lineTo(t.x, t.y!);
      ctx.strokeStyle = dim
        ? "rgba(255,255,255,0.04)"
        : isHighlighted
        ? EDGE_TYPE_COLORS[link.type] || "#444"
        : "rgba(255,255,255,0.1)";
      ctx.lineWidth = isHighlighted ? 1.5 : 0.5;
      ctx.stroke();

      // Arrow
      if (isHighlighted || !selectedId) {
        const dx = t.x - s.x;
        const dy = t.y! - s.y!;
        const len = Math.sqrt(dx * dx + dy * dy);
        if (len > 0) {
          const ux = dx / len;
          const uy = dy / len;
          const r = getNodeRadius(t);
          const ax = t.x - ux * (r + 4);
          const ay = t.y! - uy * (r + 4);
          const aSize = isHighlighted ? 5 : 3;
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(ax - ux * aSize + uy * aSize * 0.5, ay - uy * aSize - ux * aSize * 0.5);
          ctx.lineTo(ax - ux * aSize - uy * aSize * 0.5, ay - uy * aSize + ux * aSize * 0.5);
          ctx.closePath();
          ctx.fillStyle = ctx.strokeStyle;
          ctx.fill();
        }
      }
    }

    // Draw nodes
    for (const node of simNodes) {
      if (node.x == null) continue;
      const r = getNodeRadius(node);
      const isSelected = node.id === selectedId;
      const isHovered = node.id === hoveredId;
      const dim = selectedId && !highlightIds.has(node.id);

      ctx.beginPath();
      ctx.arc(node.x, node.y!, r, 0, Math.PI * 2);
      ctx.fillStyle = dim
        ? `${CATEGORY_COLORS[node.category]}33`
        : CATEGORY_COLORS[node.category];
      ctx.fill();

      if (isSelected || isHovered) {
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Label
      if (transform.k > 0.5 || isSelected || isHovered || node.core) {
        ctx.fillStyle = dim ? "rgba(255,255,255,0.15)" : "rgba(255,255,255,0.85)";
        ctx.font = `${isSelected ? "bold " : ""}${10 / Math.max(transform.k, 0.6)}px system-ui`;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(node.label, node.x, node.y! + r + 3);
      }
    }

    ctx.restore();
  }, [simNodes, simLinks, transform, selectedNode, hoveredNode, dimensions, edges]);

  function getNodeRadius(node: SimNode): number {
    const degree = edges.filter(
      (e) =>
        e.source === node.id ||
        e.target === node.id ||
        (e.source as unknown as SimNode)?.id === node.id ||
        (e.target as unknown as SimNode)?.id === node.id
    ).length;
    const base = node.core ? 7 : 5;
    return base + Math.sqrt(degree) * 1.5;
  }

  // Hit test
  const hitTest = useCallback(
    (mx: number, my: number): SimNode | null => {
      const x = (mx - transform.x) / transform.k;
      const y = (my - transform.y) / transform.k;
      for (let i = simNodes.length - 1; i >= 0; i--) {
        const n = simNodes[i];
        if (n.x == null) continue;
        const r = getNodeRadius(n) + 4;
        if ((n.x - x) ** 2 + (n.y! - y) ** 2 < r * r) return n;
      }
      return null;
    },
    [simNodes, transform]
  );

  // Mouse handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const hit = hitTest(mx, my);
    if (hit) {
      setDragging(hit);
      hit.fx = hit.x;
      hit.fy = hit.y;
      simRef.current?.alphaTarget(0.3).restart();
    } else {
      setPanning(true);
      setPanStart({ x: e.clientX - transform.x, y: e.clientY - transform.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    if (dragging) {
      dragging.fx = (mx - transform.x) / transform.k;
      dragging.fy = (my - transform.y) / transform.k;
    } else if (panning) {
      setTransform((t) => ({
        ...t,
        x: e.clientX - panStart.x,
        y: e.clientY - panStart.y,
      }));
    } else {
      setHoveredNode(hitTest(mx, my));
    }
  };

  const handleMouseUp = () => {
    if (dragging) {
      dragging.fx = null;
      dragging.fy = null;
      simRef.current?.alphaTarget(0);
      // If didn't move much, treat as click
      onSelectNode(dragging);
      setDragging(null);
    }
    setPanning(false);
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const rect = canvasRef.current!.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    setTransform((t) => ({
      x: mx - (mx - t.x) * factor,
      y: my - (my - t.y) * factor,
      k: Math.max(0.1, Math.min(4, t.k * factor)),
    }));
  };

  return (
    <canvas
      ref={canvasRef}
      className="w-full h-full cursor-grab active:cursor-grabbing"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onWheel={handleWheel}
    />
  );
}
