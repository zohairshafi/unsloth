// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Background,
  BackgroundVariant,
  Controls,
  MarkerType,
  type Edge,
  type Node,
  ReactFlow,
} from "@xyflow/react";
import dagre from "@dagrejs/dagre";
import { useCallback, useEffect, useMemo, useState } from "react";
import "@xyflow/react/dist/style.css";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { authFetch } from "@/features/auth";
import { cn } from "@/lib/utils";
import { ChevronDown } from "lucide-react";
import { toast } from "sonner";

type WikiDataKind = "source" | "analysis" | "entity" | "concept";

type WikiDataGraphNode = {
  id: string;
  kind: WikiDataKind;
  label: string;
  inbound_links: number;
  outbound_links: number;
};

type WikiDataGraphEdge = {
  id: string;
  source: string;
  target: string;
};

type WikiDataGraphResponse = {
  status: string;
  nodes: WikiDataGraphNode[];
  edges: WikiDataGraphEdge[];
};

type WikiDeleteResponse = {
  status: string;
  dry_run: boolean;
  hard_delete: boolean;
  entry_type: string;
  cascade_orphan_knowledge: boolean;
  requested_entries: string[];
  resolved_entries: string[];
  missing_entries: string[];
  invalid_entries: string[];
  planned_source_pages: string[];
  planned_analysis_pages: string[];
  planned_entity_pages: string[];
  planned_concept_pages: string[];
  planned_total_pages: number;
  archived_pages: string[];
  deleted_pages: string[];
  errors: string[];
};

interface WikiDataDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

type WikiDataNodeFilters = Record<WikiDataKind, boolean>;
type WikiGraphInteractionMode = "delete" | "explore";
type WikiGraphLayoutMode =
  | "force"
  | "kind-columns"
  | "dagre-vertical"
  | "dagre-horizontal"
  | "radial";
type WikiDataPanelSectionKey =
  | "searchFilters"
  | "filteredNodes"
  | "selection"
  | "deletePreview";

const KIND_ORDER: WikiDataKind[] = ["source", "analysis", "entity", "concept"];

const KIND_LABEL: Record<WikiDataKind, string> = {
  source: "Source",
  analysis: "Analysis",
  entity: "Entity",
  concept: "Concept",
};

const KIND_STYLE: Record<
  WikiDataKind,
  {
    border: string;
    background: string;
  }
> = {
  source: {
    border: "#5aa9f5",
    background: "#ecf7ff",
  },
  analysis: {
    border: "#9b8cff",
    background: "#f4f1ff",
  },
  entity: {
    border: "#38b86f",
    background: "#e9faef",
  },
  concept: {
    border: "#f1a756",
    background: "#fff6ec",
  },
};

const DEFAULT_KIND_FILTERS: WikiDataNodeFilters = {
  source: true,
  analysis: true,
  entity: true,
  concept: true,
};

const GRAPH_LAYOUT_OPTIONS: Array<{ mode: WikiGraphLayoutMode; label: string }> = [
  { mode: "force", label: "Force (spring-like)" },
  { mode: "kind-columns", label: "Columns by Node Type" },
  { mode: "dagre-vertical", label: "Dagre Vertical" },
  { mode: "dagre-horizontal", label: "Dagre Horizontal" },
  { mode: "radial", label: "Radial" },
];

const GRAPH_INTERACTION_MODE_LABEL: Record<WikiGraphInteractionMode, string> = {
  delete: "Delete Queue",
  explore: "Explore",
};

const GRAPH_NODE_WIDTH = 236;
const GRAPH_NODE_HEIGHT = 96;
const FORCE_LAYOUT_MAX_NODES = 220;

const DEFAULT_PANEL_SECTION_OPEN: Record<WikiDataPanelSectionKey, boolean> = {
  searchFilters: true,
  filteredNodes: true,
  selection: true,
  deletePreview: true,
};

function sortGraphNodesForLayout(nodes: WikiDataGraphNode[]): WikiDataGraphNode[] {
  const kindOrder: Record<WikiDataKind, number> = {
    source: 0,
    analysis: 1,
    entity: 2,
    concept: 3,
  };

  return [...nodes].sort((left, right) => {
    const kindDiff = kindOrder[left.kind] - kindOrder[right.kind];
    if (kindDiff !== 0) return kindDiff;

    const labelDiff = left.label.localeCompare(right.label, undefined, {
      sensitivity: "base",
    });
    if (labelDiff !== 0) return labelDiff;

    return left.id.localeCompare(right.id);
  });
}

function computeKindColumnLayoutPositions(
  nodes: WikiDataGraphNode[],
): Map<string, { x: number; y: number }> {
  const grouped: Record<WikiDataKind, WikiDataGraphNode[]> = {
    source: [],
    analysis: [],
    entity: [],
    concept: [],
  };

  for (const node of nodes) {
    grouped[node.kind].push(node);
  }
  for (const kind of KIND_ORDER) {
    grouped[kind].sort((a, b) => a.label.localeCompare(b.label));
  }

  const positions = new Map<string, { x: number; y: number }>();
  for (const [columnIndex, kind] of KIND_ORDER.entries()) {
    const rows = grouped[kind];
    for (const [rowIndex, node] of rows.entries()) {
      positions.set(node.id, {
        x: columnIndex * 320,
        y: rowIndex * 112,
      });
    }
  }

  return positions;
}

function computeDagreLayoutPositions(
  nodes: WikiDataGraphNode[],
  edges: WikiDataGraphEdge[],
  rankDirection: "TB" | "LR",
): Map<string, { x: number; y: number }> {
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));
  dagreGraph.setGraph({
    rankdir: rankDirection,
    ranksep: rankDirection === "TB" ? 140 : 220,
    nodesep: rankDirection === "TB" ? 90 : 120,
    edgesep: 24,
    marginx: 40,
    marginy: 40,
  });

  for (const node of nodes) {
    dagreGraph.setNode(node.id, {
      width: GRAPH_NODE_WIDTH,
      height: GRAPH_NODE_HEIGHT,
    });
  }

  const nodeIds = new Set(nodes.map((node) => node.id));
  for (const edge of edges) {
    if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) {
      continue;
    }
    dagreGraph.setEdge(edge.source, edge.target);
  }

  dagre.layout(dagreGraph);

  const positions = new Map<string, { x: number; y: number }>();
  for (const node of nodes) {
    const positioned = dagreGraph.node(node.id) as
      | { x: number; y: number }
      | undefined;
    if (!positioned) continue;

    positions.set(node.id, {
      x: positioned.x - GRAPH_NODE_WIDTH / 2,
      y: positioned.y - GRAPH_NODE_HEIGHT / 2,
    });
  }

  return positions;
}

function computeRadialLayoutPositions(
  nodes: WikiDataGraphNode[],
): Map<string, { x: number; y: number }> {
  const ordered = sortGraphNodesForLayout(nodes);
  const perRing = 14;
  const radiusStep = 240;

  const positions = new Map<string, { x: number; y: number }>();
  for (const [index, node] of ordered.entries()) {
    const ringIndex = Math.floor(index / perRing);
    const ringOffset = index % perRing;
    const ringNodeCount = Math.min(
      perRing,
      ordered.length - ringIndex * perRing,
    );
    const angle =
      (ringOffset / Math.max(1, ringNodeCount)) * Math.PI * 2 - Math.PI / 2;
    const radius = radiusStep * (ringIndex + 1);

    positions.set(node.id, {
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius,
    });
  }

  return positions;
}

function computeForceLayoutPositions(
  nodes: WikiDataGraphNode[],
  edges: WikiDataGraphEdge[],
): Map<string, { x: number; y: number }> {
  if (nodes.length <= 1) {
    return computeKindColumnLayoutPositions(nodes);
  }
  if (nodes.length > FORCE_LAYOUT_MAX_NODES) {
    return computeDagreLayoutPositions(nodes, edges, "TB");
  }

  const positions = computeRadialLayoutPositions(nodes);
  const velocities = new Map<string, { x: number; y: number }>();
  for (const node of nodes) {
    velocities.set(node.id, { x: 0, y: 0 });
  }

  const nodeIds = [...positions.keys()];
  const nodeIdSet = new Set(nodeIds);
  const edgePairs = edges
    .filter(
      (edge) => nodeIdSet.has(edge.source) && nodeIdSet.has(edge.target),
    )
    .map((edge) => ({ source: edge.source, target: edge.target }));

  const repulsionStrength = 28_000;
  const springStrength = 0.018;
  const springLength = 250;
  const centeringStrength = 0.003;
  const damping = 0.86;
  const iterations = 84;

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const forces = new Map<string, { x: number; y: number }>();
    for (const id of nodeIds) {
      forces.set(id, { x: 0, y: 0 });
    }

    for (let leftIndex = 0; leftIndex < nodeIds.length; leftIndex += 1) {
      const leftId = nodeIds[leftIndex];
      const leftPosition = positions.get(leftId);
      if (!leftPosition) continue;

      for (
        let rightIndex = leftIndex + 1;
        rightIndex < nodeIds.length;
        rightIndex += 1
      ) {
        const rightId = nodeIds[rightIndex];
        const rightPosition = positions.get(rightId);
        if (!rightPosition) continue;

        const dx = rightPosition.x - leftPosition.x;
        const dy = rightPosition.y - leftPosition.y;
        const distanceSquared = dx * dx + dy * dy + 0.01;
        const distance = Math.sqrt(distanceSquared);
        const force = repulsionStrength / distanceSquared;
        const fx = (dx / distance) * force;
        const fy = (dy / distance) * force;

        const leftForce = forces.get(leftId);
        const rightForce = forces.get(rightId);
        if (!leftForce || !rightForce) continue;

        leftForce.x -= fx;
        leftForce.y -= fy;
        rightForce.x += fx;
        rightForce.y += fy;
      }
    }

    for (const edge of edgePairs) {
      const sourcePosition = positions.get(edge.source);
      const targetPosition = positions.get(edge.target);
      if (!sourcePosition || !targetPosition) continue;

      const dx = targetPosition.x - sourcePosition.x;
      const dy = targetPosition.y - sourcePosition.y;
      const distance = Math.max(0.1, Math.hypot(dx, dy));
      const displacement = distance - springLength;
      const force = springStrength * displacement;
      const fx = (dx / distance) * force;
      const fy = (dy / distance) * force;

      const sourceForce = forces.get(edge.source);
      const targetForce = forces.get(edge.target);
      if (!sourceForce || !targetForce) continue;

      sourceForce.x += fx;
      sourceForce.y += fy;
      targetForce.x -= fx;
      targetForce.y -= fy;
    }

    for (const id of nodeIds) {
      const position = positions.get(id);
      const velocity = velocities.get(id);
      const force = forces.get(id);
      if (!position || !velocity || !force) continue;

      force.x += -position.x * centeringStrength;
      force.y += -position.y * centeringStrength;

      velocity.x = (velocity.x + force.x) * damping;
      velocity.y = (velocity.y + force.y) * damping;

      position.x += velocity.x;
      position.y += velocity.y;
    }
  }

  let averageX = 0;
  let averageY = 0;
  for (const id of nodeIds) {
    const position = positions.get(id);
    if (!position) continue;
    averageX += position.x;
    averageY += position.y;
  }

  averageX /= Math.max(1, nodeIds.length);
  averageY /= Math.max(1, nodeIds.length);
  for (const id of nodeIds) {
    const position = positions.get(id);
    if (!position) continue;
    position.x -= averageX;
    position.y -= averageY;
  }

  return positions;
}

function computeLayoutPositions(
  mode: WikiGraphLayoutMode,
  nodes: WikiDataGraphNode[],
  edges: WikiDataGraphEdge[],
): Map<string, { x: number; y: number }> {
  if (nodes.length === 0) {
    return new Map();
  }

  switch (mode) {
    case "kind-columns":
      return computeKindColumnLayoutPositions(nodes);
    case "dagre-vertical":
      return computeDagreLayoutPositions(nodes, edges, "TB");
    case "dagre-horizontal":
      return computeDagreLayoutPositions(nodes, edges, "LR");
    case "radial":
      return computeRadialLayoutPositions(nodes);
    case "force":
      return computeForceLayoutPositions(nodes, edges);
    default:
      return computeKindColumnLayoutPositions(nodes);
  }
}

async function parseApiErrorMessage(response: Response): Promise<string> {
  try {
    const payload = (await response.clone().json()) as { detail?: unknown };
    if (typeof payload.detail === "string" && payload.detail.trim()) {
      return payload.detail;
    }
    if (payload.detail !== undefined) {
      return `Request failed (${response.status}): ${JSON.stringify(payload.detail)}`;
    }
  } catch {
    // Ignore parse failures and use the status-only fallback.
  }
  return `Request failed (${response.status})`;
}

function trimList(items: string[], maxItems = 8): string[] {
  if (items.length <= maxItems) return items;
  const remaining = items.length - maxItems;
  return [...items.slice(0, maxItems), `... +${remaining} more`];
}

function dedupeStrings(items: string[]): string[] {
  const deduped: string[] = [];
  const seen = new Set<string>();

  for (const rawItem of items) {
    const item = String(rawItem ?? "").trim();
    if (!item || seen.has(item)) continue;
    seen.add(item);
    deduped.push(item);
  }

  return deduped;
}

function aggregateDeleteReports(reports: WikiDeleteResponse[]): WikiDeleteResponse | null {
  if (!reports.length) return null;

  const plannedSource = dedupeStrings(
    reports.flatMap((report) => report.planned_source_pages ?? []),
  );
  const plannedAnalysis = dedupeStrings(
    reports.flatMap((report) => report.planned_analysis_pages ?? []),
  );
  const plannedEntity = dedupeStrings(
    reports.flatMap((report) => report.planned_entity_pages ?? []),
  );
  const plannedConcept = dedupeStrings(
    reports.flatMap((report) => report.planned_concept_pages ?? []),
  );

  const plannedAll = dedupeStrings([
    ...plannedSource,
    ...plannedAnalysis,
    ...plannedEntity,
    ...plannedConcept,
  ]);

  const allStatusesOk = reports.every((report) => String(report.status ?? "") === "ok");

  return {
    status: allStatusesOk ? "ok" : "partial",
    dry_run: true,
    hard_delete: false,
    entry_type: reports.length > 1 ? "mixed" : reports[0]?.entry_type ?? "",
    cascade_orphan_knowledge: true,
    requested_entries: dedupeStrings(
      reports.flatMap((report) => report.requested_entries ?? []),
    ),
    resolved_entries: dedupeStrings(reports.flatMap((report) => report.resolved_entries ?? [])),
    missing_entries: dedupeStrings(reports.flatMap((report) => report.missing_entries ?? [])),
    invalid_entries: dedupeStrings(reports.flatMap((report) => report.invalid_entries ?? [])),
    planned_source_pages: plannedSource,
    planned_analysis_pages: plannedAnalysis,
    planned_entity_pages: plannedEntity,
    planned_concept_pages: plannedConcept,
    planned_total_pages: plannedAll.length,
    archived_pages: [],
    deleted_pages: [],
    errors: dedupeStrings(reports.flatMap((report) => report.errors ?? [])),
  };
}

export function WikiDataDialog({ open, onOpenChange }: WikiDataDialogProps) {
  const [loadingGraph, setLoadingGraph] = useState(false);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [includeAnalysis, setIncludeAnalysis] = useState(true);
  const [interactionMode, setInteractionMode] =
    useState<WikiGraphInteractionMode>("explore");
  const [layoutMode, setLayoutMode] = useState<WikiGraphLayoutMode>("kind-columns");
  const [searchQuery, setSearchQuery] = useState("");
  const [exploreAddQuery, setExploreAddQuery] = useState("");
  const [exploreHideUnfocused, setExploreHideUnfocused] = useState(true);
  const [exploreHopDepth, setExploreHopDepth] = useState<1 | 2>(1);
  const [kindFilters, setKindFilters] = useState<WikiDataNodeFilters>(
    DEFAULT_KIND_FILTERS,
  );
  const [panelSectionOpen, setPanelSectionOpen] = useState<
    Record<WikiDataPanelSectionKey, boolean>
  >(DEFAULT_PANEL_SECTION_OPEN);
  const [graphNodes, setGraphNodes] = useState<WikiDataGraphNode[]>([]);
  const [graphEdges, setGraphEdges] = useState<WikiDataGraphEdge[]>([]);
  const [selectedNodeIds, setSelectedNodeIds] = useState<string[]>([]);
  const [exploreNodeIds, setExploreNodeIds] = useState<string[]>([]);
  const [previewReports, setPreviewReports] = useState<WikiDeleteResponse[]>([]);

  const setPanelSectionVisibility = useCallback(
    (section: WikiDataPanelSectionKey, open: boolean): void => {
      setPanelSectionOpen((current) => ({
        ...current,
        [section]: open,
      }));
    },
    [],
  );

  const loadGraph = useCallback(async (nextIncludeAnalysis: boolean): Promise<void> => {
    setLoadingGraph(true);
    try {
      const response = await authFetch(
        `/api/inference/wiki/data/graph?include_analysis=${nextIncludeAnalysis ? "true" : "false"}`,
        {
          method: "GET",
        },
      );
      if (!response.ok) {
        throw new Error(await parseApiErrorMessage(response));
      }

      const payload = (await response.json()) as WikiDataGraphResponse;
      setGraphNodes(Array.isArray(payload.nodes) ? payload.nodes : []);
      setGraphEdges(Array.isArray(payload.edges) ? payload.edges : []);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to load wiki graph data";
      toast.error("Could not load wiki data", { description: message });
      setGraphNodes([]);
      setGraphEdges([]);
    } finally {
      setLoadingGraph(false);
    }
  }, []);

  const toggleNodeSelection = useCallback((nodeId: string): void => {
    setSelectedNodeIds((current) => {
      if (current.includes(nodeId)) {
        return current.filter((item) => item !== nodeId);
      }
      return [...current, nodeId];
    });
  }, []);

  const toggleExploreSelection = useCallback((nodeId: string): void => {
    setExploreNodeIds((current) => {
      if (current.includes(nodeId)) {
        return current.filter((item) => item !== nodeId);
      }
      return [...current, nodeId];
    });
  }, []);

  const addExploreSeedNode = useCallback((nodeId: string): void => {
    setInteractionMode("explore");
    setExploreHideUnfocused(true);
    // Ensure newly selected seed is visible in graph focus even if a prior
    // text filter would otherwise hide it.
    setSearchQuery("");
    setExploreNodeIds((current) => {
      if (current.includes(nodeId)) return current;
      return [...current, nodeId];
    });
    setExploreAddQuery("");
  }, []);

  const activeSelectionNodeIds =
    interactionMode === "explore" ? exploreNodeIds : selectedNodeIds;

  const activeSelectionNodeIdSet = useMemo(() => {
    return new Set(activeSelectionNodeIds);
  }, [activeSelectionNodeIds]);

  const clearSelection = useCallback((): void => {
    if (interactionMode === "explore") {
      setExploreNodeIds([]);
      return;
    }
    setSelectedNodeIds([]);
  }, [interactionMode]);

  const toggleKindFilter = useCallback((kind: WikiDataKind): void => {
    setKindFilters((current) => ({
      ...current,
      [kind]: !current[kind],
    }));
  }, []);

  const nodeById = useMemo(() => {
    const map = new Map<string, WikiDataGraphNode>();
    for (const node of graphNodes) {
      map.set(node.id, node);
    }
    return map;
  }, [graphNodes]);

  const selectedNodes = useMemo(() => {
    const nodes: WikiDataGraphNode[] = [];
    for (const nodeId of selectedNodeIds) {
      const node = nodeById.get(nodeId);
      if (node) nodes.push(node);
    }
    return nodes;
  }, [nodeById, selectedNodeIds]);

  const exploreSelectedNodes = useMemo(() => {
    const nodes: WikiDataGraphNode[] = [];
    for (const nodeId of exploreNodeIds) {
      const node = nodeById.get(nodeId);
      if (node) nodes.push(node);
    }
    return nodes;
  }, [exploreNodeIds, nodeById]);

  const selectedEntriesByKind = useMemo(() => {
    const grouped: Record<WikiDataKind, string[]> = {
      source: [],
      analysis: [],
      entity: [],
      concept: [],
    };

    for (const node of selectedNodes) {
      grouped[node.kind].push(node.id);
    }

    return grouped;
  }, [selectedNodes]);

  const loadPreview = useCallback(async (entriesByKind: Record<WikiDataKind, string[]>): Promise<void> => {
    const kindsToPreview = KIND_ORDER.filter((kind) => entriesByKind[kind].length > 0);
    if (!kindsToPreview.length) {
      setPreviewReports([]);
      return;
    }

    setLoadingPreview(true);
    try {
      const settledResults = await Promise.allSettled(
        kindsToPreview.map(async (kind) => {
          const response = await authFetch("/api/inference/wiki/delete/preview", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              entry_type: kind,
              entries: entriesByKind[kind],
              cascade_orphan_knowledge: true,
            }),
          });
          if (!response.ok) {
            throw new Error(await parseApiErrorMessage(response));
          }
          return (await response.json()) as WikiDeleteResponse;
        }),
      );

      const successfulReports: WikiDeleteResponse[] = [];
      const failedReasons: string[] = [];

      for (const settled of settledResults) {
        if (settled.status === "fulfilled") {
          successfulReports.push(settled.value);
        } else {
          const reason =
            settled.reason instanceof Error
              ? settled.reason.message
              : "Delete preview failed";
          failedReasons.push(reason);
        }
      }

      setPreviewReports(successfulReports);
      if (failedReasons.length > 0) {
        toast.error("Some delete previews failed", {
          description: failedReasons[0],
        });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Delete preview failed";
      toast.error("Could not preview deletion", { description: message });
      setPreviewReports([]);
    } finally {
      setLoadingPreview(false);
    }
  }, []);

  const preview = useMemo(() => aggregateDeleteReports(previewReports), [previewReports]);

  const kindCounts = useMemo(() => {
    const counts: Record<WikiDataKind, number> = {
      source: 0,
      analysis: 0,
      entity: 0,
      concept: 0,
    };
    for (const node of graphNodes) {
      counts[node.kind] += 1;
    }
    return counts;
  }, [graphNodes]);

  const selectedKindCounts = useMemo(() => {
    const counts: Record<WikiDataKind, number> = {
      source: 0,
      analysis: 0,
      entity: 0,
      concept: 0,
    };
    for (const node of selectedNodes) {
      counts[node.kind] += 1;
    }
    return counts;
  }, [selectedNodes]);

  const exploreSelectedKindCounts = useMemo(() => {
    const counts: Record<WikiDataKind, number> = {
      source: 0,
      analysis: 0,
      entity: 0,
      concept: 0,
    };
    for (const node of exploreSelectedNodes) {
      counts[node.kind] += 1;
    }
    return counts;
  }, [exploreSelectedNodes]);

  const panelSelectedNodes =
    interactionMode === "explore" ? exploreSelectedNodes : selectedNodes;

  const panelSelectedKindCounts =
    interactionMode === "explore"
      ? exploreSelectedKindCounts
      : selectedKindCounts;

  const layoutLabel = useMemo(() => {
    return (
      GRAPH_LAYOUT_OPTIONS.find((option) => option.mode === layoutMode)?.label ??
      "Custom"
    );
  }, [layoutMode]);

  const normalizedSearch = searchQuery.trim().toLowerCase();
  const normalizedExploreAddQuery = exploreAddQuery.trim().toLowerCase();

  const kindFilteredGraphNodes = useMemo(() => {
    return graphNodes.filter((node) => {
      if (!includeAnalysis && node.kind === "analysis") {
        return false;
      }
      if (!kindFilters[node.kind]) {
        return false;
      }

      return true;
    });
  }, [graphNodes, includeAnalysis, kindFilters]);

  const kindFilteredNodeIds = useMemo(() => {
    const ids = new Set<string>();
    for (const node of kindFilteredGraphNodes) {
      ids.add(node.id);
    }
    return ids;
  }, [kindFilteredGraphNodes]);

  const kindFilteredGraphEdges = useMemo(() => {
    return graphEdges.filter(
      (edge) =>
        kindFilteredNodeIds.has(edge.source) && kindFilteredNodeIds.has(edge.target),
    );
  }, [graphEdges, kindFilteredNodeIds]);

  const exploreFocusNodeIds = useMemo(() => {
    const focused = new Set<string>();
    const seeds = exploreNodeIds.filter((nodeId) => kindFilteredNodeIds.has(nodeId));
    if (!seeds.length) {
      return focused;
    }

    const adjacency = new Map<string, Set<string>>();
    for (const nodeId of kindFilteredNodeIds) {
      adjacency.set(nodeId, new Set<string>());
    }

    for (const edge of kindFilteredGraphEdges) {
      adjacency.get(edge.source)?.add(edge.target);
      adjacency.get(edge.target)?.add(edge.source);
    }

    let frontier: string[] = [];
    for (const seed of seeds) {
      focused.add(seed);
      frontier.push(seed);
    }

    for (let hop = 0; hop < exploreHopDepth; hop += 1) {
      if (frontier.length === 0) break;
      const nextFrontier: string[] = [];
      for (const nodeId of frontier) {
        const neighbors = adjacency.get(nodeId);
        if (!neighbors) continue;
        for (const neighborId of neighbors) {
          if (focused.has(neighborId)) continue;
          focused.add(neighborId);
          nextFrontier.push(neighborId);
        }
      }
      frontier = nextFrontier;
    }

    return focused;
  }, [
    exploreHopDepth,
    exploreNodeIds,
    kindFilteredGraphEdges,
    kindFilteredNodeIds,
  ]);

  const exploreModeHasFocus =
    interactionMode === "explore" && exploreFocusNodeIds.size > 0;

  const searchFilteredGraphNodes = useMemo(() => {
    return kindFilteredGraphNodes.filter((node) => {
      if (!normalizedSearch) {
        return true;
      }

      const haystack = `${node.label} ${node.id}`.toLowerCase();
      return haystack.includes(normalizedSearch);
    });
  }, [kindFilteredGraphNodes, normalizedSearch]);

  const displayGraphNodes = useMemo(() => {
    if (interactionMode !== "explore") {
      return searchFilteredGraphNodes;
    }
    if (!exploreHideUnfocused || !exploreModeHasFocus) {
      return searchFilteredGraphNodes;
    }

    return searchFilteredGraphNodes.filter((node) =>
      exploreFocusNodeIds.has(node.id),
    );
  }, [
    exploreFocusNodeIds,
    interactionMode,
    exploreHideUnfocused,
    exploreModeHasFocus,
    searchFilteredGraphNodes,
  ]);

  const displayNodeIds = useMemo(() => {
    const ids = new Set<string>();
    for (const node of displayGraphNodes) {
      ids.add(node.id);
    }
    return ids;
  }, [displayGraphNodes]);

  const displayGraphEdges = useMemo(() => {
    return kindFilteredGraphEdges.filter(
      (edge) => displayNodeIds.has(edge.source) && displayNodeIds.has(edge.target),
    );
  }, [displayNodeIds, kindFilteredGraphEdges]);

  const hiddenGraphNodeCount = Math.max(
    0,
    searchFilteredGraphNodes.length - displayGraphNodes.length,
  );

  const exploreAddCandidates = useMemo(() => {
    if (interactionMode !== "explore" || !normalizedExploreAddQuery) {
      return [];
    }

    const matched = kindFilteredGraphNodes.filter((node) => {
      const haystack = `${node.label} ${node.id}`.toLowerCase();
      return haystack.includes(normalizedExploreAddQuery);
    });

    return sortGraphNodesForLayout(matched).slice(0, 24);
  }, [
    interactionMode,
    kindFilteredGraphNodes,
    normalizedExploreAddQuery,
  ]);

  const layoutSortedNodes = useMemo(
    () => sortGraphNodesForLayout(displayGraphNodes),
    [displayGraphNodes],
  );

  const layoutPositions = useMemo(() => {
    return computeLayoutPositions(layoutMode, layoutSortedNodes, displayGraphEdges);
  }, [displayGraphEdges, layoutMode, layoutSortedNodes]);

  const graphViewportKey = useMemo(
    () =>
      `${layoutMode}-${displayGraphNodes.length}-${displayGraphEdges.length}-${interactionMode}-${exploreHideUnfocused ? "hide" : "show"}-${exploreHopDepth}`,
    [
      displayGraphEdges.length,
      displayGraphNodes.length,
      exploreHideUnfocused,
      exploreHopDepth,
      interactionMode,
      layoutMode,
    ],
  );

  const selectFilteredNodes = useCallback((): void => {
    if (interactionMode === "explore") {
      setExploreNodeIds((current) =>
        dedupeStrings([...current, ...searchFilteredGraphNodes.map((node) => node.id)]),
      );
      return;
    }

    setSelectedNodeIds((current) =>
      dedupeStrings([...current, ...searchFilteredGraphNodes.map((node) => node.id)]),
    );
  }, [interactionMode, searchFilteredGraphNodes]);

  const clearFilteredNodes = useCallback((): void => {
    const searchFilteredNodeIds = new Set(
      searchFilteredGraphNodes.map((node) => node.id),
    );

    if (interactionMode === "explore") {
      setExploreNodeIds((current) =>
        current.filter((nodeId) => !searchFilteredNodeIds.has(nodeId)),
      );
      return;
    }

    setSelectedNodeIds((current) =>
      current.filter((nodeId) => !searchFilteredNodeIds.has(nodeId)),
    );
  }, [interactionMode, searchFilteredGraphNodes]);

  const flowNodes = useMemo<Node[]>(() => {
    const deleteSelectedIds = new Set(selectedNodeIds);
    const exploreSelectedIds = new Set(exploreNodeIds);

    const nodes: Node[] = [];
    for (const node of layoutSortedNodes) {
      const kind = node.kind;
      const palette = KIND_STYLE[kind];
      const deleteSelected = deleteSelectedIds.has(node.id);
      const exploreSelected = exploreSelectedIds.has(node.id);
      const inExploreFocus = exploreFocusNodeIds.has(node.id);
      const dimNode = exploreModeHasFocus && !inExploreFocus;

      const borderColor = exploreSelected
        ? "#0f172a"
        : deleteSelected
          ? "#111827"
          : inExploreFocus && interactionMode === "explore"
            ? "#475569"
            : palette.border;

      const boxShadow = exploreSelected
        ? "0 10px 30px rgba(15, 23, 42, 0.24)"
        : deleteSelected
          ? "0 8px 24px rgba(17, 24, 39, 0.16)"
          : inExploreFocus && interactionMode === "explore"
            ? "0 6px 18px rgba(51, 65, 85, 0.16)"
            : "0 3px 10px rgba(15, 23, 42, 0.08)";
      const position = layoutPositions.get(node.id) ?? { x: 0, y: 0 };

      nodes.push({
        id: node.id,
        position,
        data: {
          label: (
            <div className="space-y-1">
              <div className="text-[10px] font-semibold tracking-wide uppercase text-slate-900">
                {KIND_LABEL[kind]}
              </div>
              <div className="line-clamp-2 text-xs font-semibold text-slate-900">
                {node.label}
              </div>
              <div className="text-[11px] font-semibold text-slate-800">
                in {node.inbound_links} | out {node.outbound_links}
              </div>
            </div>
          ),
        },
        style: {
          width: GRAPH_NODE_WIDTH,
          borderRadius: 14,
          border: `2px solid ${borderColor}`,
          background: palette.background,
          boxShadow,
          opacity: dimNode ? 0.26 : 1,
          padding: 10,
        },
      });
    }

    return nodes;
  }, [
    exploreFocusNodeIds,
    exploreModeHasFocus,
    exploreNodeIds,
    interactionMode,
    layoutPositions,
    layoutSortedNodes,
    selectedNodeIds,
  ]);

  const flowEdges = useMemo<Edge[]>(() => {
    const exploreSelectedIds = new Set(exploreNodeIds);

    return displayGraphEdges.map((edge) => {
      const edgeInExploreFocus =
        exploreFocusNodeIds.has(edge.source) &&
        exploreFocusNodeIds.has(edge.target);
      const touchesExploreSelection =
        exploreSelectedIds.has(edge.source) || exploreSelectedIds.has(edge.target);

      const stroke = !exploreModeHasFocus
        ? "#94a3b8"
        : edgeInExploreFocus
          ? touchesExploreSelection
            ? "#334155"
            : "#64748b"
          : "#94a3b8";

      const strokeWidth = !exploreModeHasFocus
        ? 1.3
        : edgeInExploreFocus
          ? touchesExploreSelection
            ? 2
            : 1.6
          : 1;

      const opacity = !exploreModeHasFocus ? 1 : edgeInExploreFocus ? 0.82 : 0.12;

      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: stroke,
        },
        style: {
          stroke,
          strokeWidth,
          opacity,
        },
        animated: false,
      };
    });
  }, [displayGraphEdges, exploreFocusNodeIds, exploreModeHasFocus, exploreNodeIds]);

  useEffect(() => {
    if (!open) return;
    setInteractionMode("explore");
    setPreviewReports([]);
    setSelectedNodeIds([]);
    setExploreNodeIds([]);
    setExploreAddQuery("");
    setExploreHideUnfocused(true);
    setExploreHopDepth(1);
    void loadGraph(includeAnalysis);
  }, [includeAnalysis, loadGraph, open]);

  useEffect(() => {
    const availableNodeIds = new Set(graphNodes.map((node) => node.id));
    setSelectedNodeIds((current) => {
      const next = current.filter((nodeId) => availableNodeIds.has(nodeId));
      return next.length === current.length ? current : next;
    });

    setExploreNodeIds((current) => {
      const next = current.filter((nodeId) => availableNodeIds.has(nodeId));
      return next.length === current.length ? current : next;
    });
  }, [graphNodes]);

  useEffect(() => {
    if (!open) return;
    const selectedCount = KIND_ORDER.reduce(
      (total, kind) => total + selectedEntriesByKind[kind].length,
      0,
    );
    if (selectedCount <= 0) {
      setPreviewReports([]);
      return;
    }
    void loadPreview(selectedEntriesByKind);
  }, [loadPreview, open, selectedEntriesByKind]);

  async function handleApplyDelete(): Promise<void> {
    if (!preview || deleting) return;

    const plannedPages = Number(preview.planned_total_pages ?? 0);
    const selectedCount = selectedNodeIds.length;
    if (plannedPages <= 0) {
      toast.warning("Nothing to delete for this selection");
      return;
    }

    const confirmed = window.confirm(
      `Archive ${plannedPages} wiki page(s) from ${selectedCount} selected node(s)?`,
    );
    if (!confirmed) return;

    setDeleting(true);
    try {
      const kindsToDelete = KIND_ORDER.filter((kind) => selectedEntriesByKind[kind].length > 0);
      const settledResults = await Promise.allSettled(
        kindsToDelete.map(async (kind) => {
          const response = await authFetch("/api/inference/wiki/delete/apply", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              entry_type: kind,
              entries: selectedEntriesByKind[kind],
              cascade_orphan_knowledge: true,
              hard_delete: false,
            }),
          });
          if (!response.ok) {
            throw new Error(await parseApiErrorMessage(response));
          }
          return (await response.json()) as WikiDeleteResponse;
        }),
      );

      const successfulPayloads: WikiDeleteResponse[] = [];
      const failedReasons: string[] = [];

      for (const settled of settledResults) {
        if (settled.status === "fulfilled") {
          successfulPayloads.push(settled.value);
        } else {
          const reason =
            settled.reason instanceof Error ? settled.reason.message : "Delete apply failed";
          failedReasons.push(reason);
        }
      }

      if (!successfulPayloads.length) {
        throw new Error(failedReasons[0] ?? "Delete apply failed");
      }

      const archived = successfulPayloads.reduce(
        (total, payload) => total + (Array.isArray(payload.archived_pages) ? payload.archived_pages.length : 0),
        0,
      );
      const deleted = successfulPayloads.reduce(
        (total, payload) => total + (Array.isArray(payload.deleted_pages) ? payload.deleted_pages.length : 0),
        0,
      );

      if (failedReasons.length > 0) {
        toast.warning("Wiki data partially updated", {
          description: `Archived: ${archived}, hard deleted: ${deleted}. First error: ${failedReasons[0]}`,
        });
      } else {
        toast.success("Wiki data updated", {
          description: `Archived: ${archived}, hard deleted: ${deleted}`,
        });
      }

      setSelectedNodeIds([]);
      setPreviewReports([]);
      await loadGraph(includeAnalysis);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Delete apply failed";
      toast.error("Could not delete wiki entry", { description: message });
    } finally {
      setDeleting(false);
    }
  }

  const canDelete =
    interactionMode === "delete" &&
    selectedNodeIds.length > 0 &&
    Boolean(preview) &&
    Number(preview?.planned_total_pages ?? 0) > 0 &&
    !loadingPreview &&
    !deleting;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-7xl gap-4 p-0 sm:max-w-[min(98vw,1440px)]">
        <DialogHeader className="border-b border-border/70 px-6 pt-5 pb-4">
          <DialogTitle className="font-heading text-lg">View/Edit Wiki Data</DialogTitle>
          <DialogDescription>
            Choose Delete Queue mode to queue archives, or Explore mode to inspect connectivity.
          </DialogDescription>
          <div className="flex flex-wrap items-center gap-2 pt-2 text-xs text-muted-foreground">
            <Badge variant="outline">Nodes: {graphNodes.length}</Badge>
            <Badge variant="outline">Visible: {displayGraphNodes.length}</Badge>
            <Badge variant="outline">Edges: {displayGraphEdges.length}/{graphEdges.length}</Badge>
            {interactionMode === "explore" && hiddenGraphNodeCount > 0 ? (
              <Badge variant="outline">Hidden: {hiddenGraphNodeCount}</Badge>
            ) : null}
            <Badge variant="outline">Mode: {GRAPH_INTERACTION_MODE_LABEL[interactionMode]}</Badge>
            <Badge variant="outline">Delete Queue: {selectedNodeIds.length}</Badge>
            <Badge variant="outline">Explore Selected: {exploreNodeIds.length}</Badge>
            {interactionMode === "explore" ? (
              <Badge variant="outline">Focus Depth: {exploreHopDepth}-hop</Badge>
            ) : null}
            <Badge variant="outline">Layout: {layoutLabel}</Badge>
            <Badge variant="outline">Sources: {kindCounts.source}</Badge>
            {includeAnalysis ? <Badge variant="outline">Analyses: {kindCounts.analysis}</Badge> : null}
            <Badge variant="outline">Entities: {kindCounts.entity}</Badge>
            <Badge variant="outline">Concepts: {kindCounts.concept}</Badge>
          </div>
          <div className="flex flex-wrap items-center gap-2 pt-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setInteractionMode("delete")}
              disabled={deleting}
              className={cn(
                interactionMode === "delete" &&
                  "border-foreground/60 bg-foreground/5 text-foreground",
              )}
            >
              Delete Queue Mode
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setInteractionMode("explore")}
              disabled={deleting}
              className={cn(
                interactionMode === "explore" &&
                  "border-foreground/60 bg-foreground/5 text-foreground",
              )}
            >
              Explore Mode
            </Button>
            <span className="text-xs text-muted-foreground">
              {interactionMode === "explore"
                ? `Explore mode focuses ${exploreHopDepth}-hop neighbors; hide/show and add-seed controls are in Search & Filters.`
                : "Delete Queue mode selects nodes for archive preview and apply."}
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-2 pt-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setIncludeAnalysis((value) => !value)}
              disabled={loadingGraph || deleting}
            >
              {includeAnalysis ? "Hide analysis nodes" : "Include analysis nodes"}
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => toggleKindFilter("source")}
              disabled={loadingGraph || deleting}
            >
              {kindFilters.source ? "Hide source nodes" : "Show source nodes"}
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => toggleKindFilter("entity")}
              disabled={loadingGraph || deleting}
            >
              {kindFilters.entity ? "Hide entity nodes" : "Show entity nodes"}
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => toggleKindFilter("concept")}
              disabled={loadingGraph || deleting}
            >
              {kindFilters.concept ? "Hide concept nodes" : "Show concept nodes"}
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => void loadGraph(includeAnalysis)}
              disabled={loadingGraph || deleting}
            >
              {loadingGraph ? "Refreshing..." : "Refresh Graph"}
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={clearSelection}
              disabled={activeSelectionNodeIds.length === 0 || deleting}
            >
              {interactionMode === "explore" ? "Clear Explore Selection" : "Clear Selection"}
            </Button>
          </div>
          <div className="flex flex-wrap items-center gap-2 pt-2">
            {GRAPH_LAYOUT_OPTIONS.map((option) => (
              <Button
                key={`layout-${option.mode}`}
                type="button"
                variant="outline"
                size="sm"
                onClick={() => setLayoutMode(option.mode)}
                disabled={deleting}
                className={cn(
                  layoutMode === option.mode &&
                    "border-foreground/60 bg-foreground/5 text-foreground",
                )}
              >
                {option.label}
              </Button>
            ))}
          </div>
        </DialogHeader>

        <div className="grid gap-4 px-6 pb-2 lg:grid-cols-[minmax(0,1fr)_420px]">
          <div className="rounded-2xl border border-border/70 bg-background/70">
            {loadingGraph ? (
              <div className="flex h-[min(62vh,720px)] items-center justify-center text-sm text-muted-foreground">
                Loading wiki graph...
              </div>
            ) : flowNodes.length === 0 ? (
              <div className="flex h-[min(62vh,720px)] items-center justify-center text-sm text-muted-foreground">
                {graphNodes.length <= 0
                  ? "No wiki data nodes found."
                  : interactionMode === "explore" &&
                      exploreHideUnfocused &&
                      exploreModeHasFocus
                    ? `No nodes remain in ${exploreHopDepth}-hop focus for the current search/filter; add another seed node.`
                    : "No nodes match current filters."}
              </div>
            ) : (
              <div className="h-[min(62vh,720px)]">
                <ReactFlow
                  key={graphViewportKey}
                  nodes={flowNodes}
                  edges={flowEdges}
                  fitView
                  fitViewOptions={{ padding: 0.35, minZoom: 0.02, maxZoom: 1.2 }}
                  minZoom={0.02}
                  maxZoom={2.4}
                  onNodeClick={(_event, node) => {
                    const nodeId = String(node.id);
                    if (interactionMode === "explore") {
                      toggleExploreSelection(nodeId);
                      return;
                    }
                    toggleNodeSelection(nodeId);
                  }}
                  nodesConnectable={false}
                  panOnDrag
                >
                  <Controls showInteractive={false} />
                  <Background variant={BackgroundVariant.Dots} gap={18} size={1.2} />
                </ReactFlow>
              </div>
            )}
          </div>

          <ScrollArea className="h-[min(62vh,720px)] rounded-2xl border border-border/70 bg-background/70 p-4">
            <div className="space-y-3">
              <Collapsible
                open={panelSectionOpen.searchFilters}
                onOpenChange={(open) => setPanelSectionVisibility("searchFilters", open)}
                className="rounded-xl border border-border/70 bg-background/60 px-3 py-2"
              >
                <CollapsibleTrigger className="flex w-full items-center justify-between text-left">
                  <h3 className="text-sm font-semibold tracking-wide text-foreground/90">
                    Search & Filters
                  </h3>
                  <ChevronDown
                    className={cn(
                      "size-4 text-muted-foreground transition-transform duration-200",
                      panelSectionOpen.searchFilters ? "rotate-0" : "-rotate-90",
                    )}
                  />
                </CollapsibleTrigger>
                <CollapsibleContent className="pt-2">
                  <div className="space-y-2">
                    <Input
                      value={searchQuery}
                      onChange={(event) => setSearchQuery(event.target.value)}
                      placeholder="Filter by node label or id"
                      disabled={loadingGraph || deleting}
                    />
                    <div className="flex flex-wrap items-center gap-2">
                      {KIND_ORDER.map((kind) => (
                        <Button
                          key={`kind-filter-${kind}`}
                          type="button"
                          variant="outline"
                          size="sm"
                          onClick={() => toggleKindFilter(kind)}
                          className={cn(!kindFilters[kind] && "opacity-55")}
                          disabled={deleting}
                        >
                          {KIND_LABEL[kind]}
                        </Button>
                      ))}
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={selectFilteredNodes}
                        disabled={searchFilteredGraphNodes.length === 0 || deleting}
                      >
                        {interactionMode === "explore"
                          ? "Select Filtered (Explore)"
                          : "Select Filtered"}
                      </Button>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={clearFilteredNodes}
                        disabled={activeSelectionNodeIds.length === 0 || deleting}
                      >
                        {interactionMode === "explore"
                          ? "Clear Filtered (Explore)"
                          : "Clear Filtered"}
                      </Button>
                    </div>
                    {interactionMode === "explore" ? (
                      <div className="space-y-2 rounded-lg border border-border/70 bg-muted/20 p-2">
                        <div className="flex flex-wrap items-center gap-2">
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            onClick={() => setExploreHopDepth(1)}
                            disabled={deleting}
                            className={cn(
                              exploreHopDepth === 1 &&
                                "border-foreground/60 bg-foreground/5 text-foreground",
                            )}
                          >
                            1-Hop Focus
                          </Button>
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            onClick={() => setExploreHopDepth(2)}
                            disabled={deleting}
                            className={cn(
                              exploreHopDepth === 2 &&
                                "border-foreground/60 bg-foreground/5 text-foreground",
                            )}
                          >
                            2-Hop Focus
                          </Button>
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            onClick={() => setExploreHideUnfocused((value) => !value)}
                            disabled={deleting}
                          >
                            {exploreHideUnfocused
                              ? "Show All Filtered Nodes"
                              : `Hide Non-${exploreHopDepth}-Hop Nodes`}
                          </Button>
                          <span className="text-[11px] text-muted-foreground">
                            {exploreHideUnfocused
                              ? `${hiddenGraphNodeCount} node(s) hidden by current ${exploreHopDepth}-hop focus.`
                              : "All filtered nodes are currently visible."}
                          </span>
                        </div>
                        <Input
                          value={exploreAddQuery}
                          onChange={(event) => setExploreAddQuery(event.target.value)}
                          onKeyDown={(event) => {
                            if (event.key !== "Enter") return;
                            event.preventDefault();
                            const first = exploreAddCandidates[0];
                            if (!first) return;
                            addExploreSeedNode(first.id);
                          }}
                          placeholder="Add node to explore seeds (search all filtered nodes)"
                          disabled={loadingGraph || deleting}
                        />
                        {normalizedExploreAddQuery ? (
                          exploreAddCandidates.length > 0 ? (
                            <div className="max-h-36 space-y-1 overflow-auto rounded-lg border border-border/70 p-2">
                              {exploreAddCandidates.map((node) => {
                                const alreadySeeded = exploreNodeIds.includes(node.id);
                                return (
                                  <div
                                    key={`explore-candidate-${node.id}`}
                                    className={cn(
                                      "flex items-start justify-between gap-2 rounded-md border border-transparent px-1 py-1",
                                      "cursor-pointer hover:border-foreground/15 hover:bg-muted/50",
                                    )}
                                    role="button"
                                    tabIndex={0}
                                    onClick={() => addExploreSeedNode(node.id)}
                                    onKeyDown={(event) => {
                                      if (event.key === "Enter" || event.key === " ") {
                                        event.preventDefault();
                                        addExploreSeedNode(node.id);
                                      }
                                    }}
                                  >
                                    <div className="min-w-0 text-xs">
                                      <p className="truncate font-medium text-foreground">
                                        {node.label}
                                      </p>
                                      <p className="truncate text-muted-foreground">
                                        {node.id}
                                      </p>
                                    </div>
                                    <Button
                                      type="button"
                                      variant="outline"
                                      size="sm"
                                      disabled={deleting}
                                      onClick={() => addExploreSeedNode(node.id)}
                                    >
                                      {alreadySeeded ? "Focus" : "Select"}
                                    </Button>
                                  </div>
                                );
                              })}
                            </div>
                          ) : (
                            <p className="text-[11px] text-muted-foreground">
                              No matching nodes found for that seed query.
                            </p>
                          )
                        ) : (
                          <p className="text-[11px] text-muted-foreground">
                            Search for hidden nodes here to add them to the explore seed set.
                          </p>
                        )}
                      </div>
                    ) : null}
                    <p className="text-xs text-muted-foreground">
                      Showing {searchFilteredGraphNodes.length} nodes for current filters.
                      {interactionMode === "explore"
                        ? ` ${displayGraphNodes.length} shown in graph.`
                        : ""}
                      {interactionMode === "explore" && hiddenGraphNodeCount > 0
                        ? ` ${hiddenGraphNodeCount} hidden by ${exploreHopDepth}-hop explore focus.`
                        : ""}
                    </p>
                  </div>
                </CollapsibleContent>
              </Collapsible>

              <Collapsible
                open={panelSectionOpen.filteredNodes}
                onOpenChange={(open) => setPanelSectionVisibility("filteredNodes", open)}
                className="rounded-xl border border-border/70 bg-background/60 px-3 py-2"
              >
                <CollapsibleTrigger className="flex w-full items-center justify-between text-left">
                  <h3 className="text-sm font-semibold tracking-wide text-foreground/90">
                    Filtered Nodes
                  </h3>
                  <ChevronDown
                    className={cn(
                      "size-4 text-muted-foreground transition-transform duration-200",
                      panelSectionOpen.filteredNodes ? "rotate-0" : "-rotate-90",
                    )}
                  />
                </CollapsibleTrigger>
                <CollapsibleContent className="pt-2">
                  {searchFilteredGraphNodes.length === 0 ? (
                    <p className="text-xs text-muted-foreground">
                      No nodes available for current filters.
                    </p>
                  ) : (
                    <div className="max-h-56 space-y-1 overflow-auto rounded-xl border border-border/70 p-2">
                      {searchFilteredGraphNodes.slice(0, 200).map((node) => (
                        <label
                          key={`selectable-${node.id}`}
                          className={cn(
                            "flex cursor-pointer items-start gap-2 rounded-lg border border-transparent px-2 py-1.5 text-xs",
                            activeSelectionNodeIdSet.has(node.id) &&
                              "border-foreground/20 bg-muted/60",
                          )}
                        >
                          <input
                            type="checkbox"
                            className="mt-0.5 h-4 w-4"
                            checked={activeSelectionNodeIdSet.has(node.id)}
                            onChange={() => {
                              if (interactionMode === "explore") {
                                toggleExploreSelection(node.id);
                                return;
                              }
                              toggleNodeSelection(node.id);
                            }}
                          />
                          <span className="min-w-0">
                            <span className="block truncate font-medium text-foreground">
                              {node.label}
                            </span>
                            <span className="block truncate text-muted-foreground">
                              {node.id}
                            </span>
                          </span>
                        </label>
                      ))}
                      {searchFilteredGraphNodes.length > 200 ? (
                        <p className="px-2 pt-1 text-[11px] text-muted-foreground">
                          Showing first 200 nodes. Refine filters to narrow the list.
                        </p>
                      ) : null}
                    </div>
                  )}
                </CollapsibleContent>
              </Collapsible>

              <Collapsible
                open={panelSectionOpen.selection}
                onOpenChange={(open) => setPanelSectionVisibility("selection", open)}
                className="rounded-xl border border-border/70 bg-background/60 px-3 py-2"
              >
                <CollapsibleTrigger className="flex w-full items-center justify-between text-left">
                  <h3 className="text-sm font-semibold tracking-wide text-foreground/90">
                    {interactionMode === "explore" ? "Explore Selection" : "Selection"}
                  </h3>
                  <ChevronDown
                    className={cn(
                      "size-4 text-muted-foreground transition-transform duration-200",
                      panelSectionOpen.selection ? "rotate-0" : "-rotate-90",
                    )}
                  />
                </CollapsibleTrigger>
                <CollapsibleContent className="pt-2">
                  {panelSelectedNodes.length === 0 ? (
                    <p className="text-xs text-muted-foreground">
                      {interactionMode === "explore"
                        ? `Select one or more nodes to highlight ${exploreHopDepth}-hop neighbors.`
                        : "Select one or more nodes from the graph or filtered list."}
                    </p>
                  ) : (
                    <div className="space-y-2">
                      <div className="flex flex-wrap items-center gap-2 text-xs">
                        <Badge variant="outline">Selected: {panelSelectedNodes.length}</Badge>
                        {panelSelectedKindCounts.source > 0 ? (
                          <Badge variant="outline">Sources: {panelSelectedKindCounts.source}</Badge>
                        ) : null}
                        {panelSelectedKindCounts.analysis > 0 ? (
                          <Badge variant="outline">Analyses: {panelSelectedKindCounts.analysis}</Badge>
                        ) : null}
                        {panelSelectedKindCounts.entity > 0 ? (
                          <Badge variant="outline">Entities: {panelSelectedKindCounts.entity}</Badge>
                        ) : null}
                        {panelSelectedKindCounts.concept > 0 ? (
                          <Badge variant="outline">Concepts: {panelSelectedKindCounts.concept}</Badge>
                        ) : null}
                        {interactionMode === "explore" ? (
                          <Badge variant="outline">
                            {exploreHopDepth}-Hop Focus: {exploreFocusNodeIds.size}
                          </Badge>
                        ) : null}
                      </div>

                      <div className="max-h-44 space-y-1 overflow-auto rounded-xl border border-border/70 p-2">
                        {panelSelectedNodes.slice(0, 40).map((node) => (
                          <div
                            key={`selected-${node.id}`}
                            className={cn(
                              "flex items-start justify-between gap-2 rounded-lg border px-2 py-1.5",
                              node.kind === "source" && "border-sky-300/60 bg-sky-50/40",
                              node.kind === "analysis" && "border-violet-300/60 bg-violet-50/40",
                              node.kind === "entity" && "border-emerald-300/60 bg-emerald-50/40",
                              node.kind === "concept" && "border-amber-300/60 bg-amber-50/40",
                            )}
                          >
                            <div className="min-w-0 text-xs">
                              <p className="truncate font-medium text-foreground">{node.label}</p>
                              <p className="truncate text-muted-foreground">{node.id}</p>
                            </div>
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              onClick={() => {
                                if (interactionMode === "explore") {
                                  toggleExploreSelection(node.id);
                                  return;
                                }
                                toggleNodeSelection(node.id);
                              }}
                              disabled={deleting}
                            >
                              Remove
                            </Button>
                          </div>
                        ))}
                        {panelSelectedNodes.length > 40 ? (
                          <p className="px-1 pt-1 text-[11px] text-muted-foreground">
                            ... +{panelSelectedNodes.length - 40} more selected nodes
                          </p>
                        ) : null}
                      </div>
                    </div>
                  )}
                </CollapsibleContent>
              </Collapsible>

              <Collapsible
                open={panelSectionOpen.deletePreview}
                onOpenChange={(open) => setPanelSectionVisibility("deletePreview", open)}
                className="rounded-xl border border-border/70 bg-background/60 px-3 py-2"
              >
                <CollapsibleTrigger className="flex w-full items-center justify-between text-left">
                  <h3 className="text-sm font-semibold tracking-wide text-foreground/90">
                    Delete Preview
                  </h3>
                  <ChevronDown
                    className={cn(
                      "size-4 text-muted-foreground transition-transform duration-200",
                      panelSectionOpen.deletePreview ? "rotate-0" : "-rotate-90",
                    )}
                  />
                </CollapsibleTrigger>
                <CollapsibleContent className="pt-2">
                  {loadingPreview ? (
                    <p className="text-xs text-muted-foreground">Calculating delete impact...</p>
                  ) : !preview ? (
                    <p className="text-xs text-muted-foreground">
                      {interactionMode === "explore"
                        ? "Explore mode is active. Switch to Delete Queue mode and select nodes to preview deletion impact."
                        : "Select at least one node to preview deletion impact."}
                    </p>
                  ) : (
                    <div className="space-y-3 text-xs">
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge variant="outline">Planned pages: {preview.planned_total_pages}</Badge>
                        <Badge variant="outline">Status: {preview.status}</Badge>
                        <Badge variant="outline">Preview groups: {previewReports.length}</Badge>
                      </div>

                      {preview.missing_entries.length > 0 ? (
                        <div>
                          <p className="font-semibold text-foreground/90">Missing</p>
                          <ul className="pt-1 text-muted-foreground">
                            {trimList(preview.missing_entries).map((item) => (
                              <li key={`missing-${item}`}>{item}</li>
                            ))}
                          </ul>
                        </div>
                      ) : null}

                      {preview.invalid_entries.length > 0 ? (
                        <div>
                          <p className="font-semibold text-foreground/90">Invalid</p>
                          <ul className="pt-1 text-muted-foreground">
                            {trimList(preview.invalid_entries).map((item) => (
                              <li key={`invalid-${item}`}>{item}</li>
                            ))}
                          </ul>
                        </div>
                      ) : null}

                      {preview.errors.length > 0 ? (
                        <div>
                          <p className="font-semibold text-foreground/90">Errors</p>
                          <ul className="pt-1 text-muted-foreground">
                            {trimList(preview.errors).map((item) => (
                              <li key={`error-${item}`}>{item}</li>
                            ))}
                          </ul>
                        </div>
                      ) : null}

                      {preview.planned_source_pages.length > 0 ? (
                        <div>
                          <p className="font-semibold text-foreground/90">Sources</p>
                          <ul className="pt-1 text-muted-foreground">
                            {trimList(preview.planned_source_pages).map((item) => (
                              <li key={`source-${item}`}>{item}</li>
                            ))}
                          </ul>
                        </div>
                      ) : null}

                      {preview.planned_analysis_pages.length > 0 ? (
                        <div>
                          <p className="font-semibold text-foreground/90">Analyses</p>
                          <ul className="pt-1 text-muted-foreground">
                            {trimList(preview.planned_analysis_pages).map((item) => (
                              <li key={`analysis-${item}`}>{item}</li>
                            ))}
                          </ul>
                        </div>
                      ) : null}

                      {preview.planned_entity_pages.length > 0 ? (
                        <div>
                          <p className="font-semibold text-foreground/90">Entities</p>
                          <ul className="pt-1 text-muted-foreground">
                            {trimList(preview.planned_entity_pages).map((item) => (
                              <li key={`entity-${item}`}>{item}</li>
                            ))}
                          </ul>
                        </div>
                      ) : null}

                      {preview.planned_concept_pages.length > 0 ? (
                        <div>
                          <p className="font-semibold text-foreground/90">Concepts</p>
                          <ul className="pt-1 text-muted-foreground">
                            {trimList(preview.planned_concept_pages).map((item) => (
                              <li key={`concept-${item}`}>{item}</li>
                            ))}
                          </ul>
                        </div>
                      ) : null}
                    </div>
                  )}
                </CollapsibleContent>
              </Collapsible>
            </div>
          </ScrollArea>
        </div>

        <DialogFooter className="border-t border-border/70 px-6 py-4 sm:justify-between">
          <p className="text-xs text-muted-foreground">
            Deletions archive files into wiki/.archive by default.
          </p>
          <div className="flex items-center gap-2">
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
              Close
            </Button>
            <Button
              type="button"
              variant="dark"
              onClick={() => void handleApplyDelete()}
              disabled={!canDelete}
            >
              {interactionMode === "explore"
                ? "Switch to Delete Queue Mode"
                : deleting
                  ? "Deleting..."
                  : "Delete Selected Nodes"}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
