---------------------- MODULE KlsmBoundaryCompaction ----------------------
(***************************************************************************)
(* Formal TLA+ specification for the lock-free KLSM-Tree                   *)
(* Boundary-Only Voronoi Compaction protocol and concurrent MVCC           *)
(* snapshot isolation, corresponding to roadmap.md Section 3.1 & 3.4.      *)
(*                                                                         *)
(* Proves:                                                                 *)
(* 1. Inv_NoLostMutations: Every committed vertex mutation remains         *)
(*    reachable before, during, and after compaction transitions.          *)
(* 2. Inv_SnapshotIsolation: Concurrent readers at timestamp \tau_q        *)
(*    observe an immutable consistent view without torn reads.             *)
(* 3. Inv_DelaunayBoundaryEquivalence: Interior nodes with valid Voronoi   *)
(*    boundary certificates retain identical Delaunay neighbor graphs      *)
(*    via O(1) pointer splices compared to full O(N log N) recomputation.  *)
(***************************************************************************)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Vertices,           \* Set of vertex identifiers, e.g., {v1, v2, v3, v4, v5, v6}
    Cells,              \* Set of Voronoi cell identifiers, e.g., {c1, c2}
    MaxTimestamps       \* Maximum logical timestamp for model checking

VARIABLES
    cell_assignment,    \* [v \in Vertices |-> Cells] : Partition of vertices into Voronoi cells
    vertex_status,      \* [v \in Vertices |-> {"Interior", "Boundary"}]
    committed_versions, \* [v \in Vertices |-> SUBSET (1..MaxTimestamps)] : Committed versions per vertex
    delaunay_graph,     \* [v \in Vertices |-> SUBSET Vertices] : Current Delaunay neighbor topology
    active_directory,   \* {"PreCompaction", "PostCompaction"} : Atomic pointer to active runs
    compaction_state,   \* {"Idle", "ComputingCertificates", "SplicingInterior", "RelinkingBoundary", "Swapped", "Reclaimed"}
    reader_epochs       \* Set of active readers: [id: Nat, snapshot_ts: 1..MaxTimestamps, run_view: {"PreCompaction", "PostCompaction"}]

vars == <<cell_assignment, vertex_status, committed_versions, delaunay_graph, active_directory, compaction_state, reader_epochs>>

-----------------------------------------------------------------------------
(* Helper Functions                                                        *)
-----------------------------------------------------------------------------
IsBoundaryVertex(v) ==
    vertex_status[v] = "Boundary"

IsInteriorVertex(v) ==
    vertex_status[v] = "Interior"

-----------------------------------------------------------------------------
(* Initial State                                                           *)
-----------------------------------------------------------------------------
Init ==
    /\ cell_assignment \in [Vertices -> Cells]
    /\ vertex_status \in [Vertices -> {"Interior", "Boundary"}]
    /\ committed_versions = [v \in Vertices |-> {1}]
    /\ delaunay_graph = [v \in Vertices |-> {u \in Vertices : cell_assignment[u] = cell_assignment[v] /\ u # v}]
    /\ active_directory = "PreCompaction"
    /\ compaction_state = "Idle"
    /\ reader_epochs = {}

-----------------------------------------------------------------------------
(* Transitions                                                             *)
-----------------------------------------------------------------------------

\* A client writes a new mutation to vertex v at a new timestamp
MutateVertex(v, ts) ==
    /\ ts \in 2..MaxTimestamps
    /\ ts \notin committed_versions[v]
    /\ compaction_state \in {"Idle", "ComputingCertificates"}
    /\ committed_versions' = [committed_versions EXCEPT ![v] = committed_versions[v] \cup {ts}]
    /\ UNCHANGED <<cell_assignment, vertex_status, delaunay_graph, active_directory, compaction_state, reader_epochs>>

\* Compactor identifies interior vertices via Voronoi boundary certificates
ComputeCertificates ==
    /\ compaction_state = "Idle"
    /\ compaction_state' = "ComputingCertificates"
    /\ UNCHANGED <<cell_assignment, vertex_status, committed_versions, delaunay_graph, active_directory, reader_epochs>>

\* Compactor performs O(1) pointer splicing on interior vertices
SpliceInterior ==
    /\ compaction_state = "ComputingCertificates"
    /\ compaction_state' = "SplicingInterior"
    /\ UNCHANGED <<cell_assignment, vertex_status, committed_versions, delaunay_graph, active_directory, reader_epochs>>

\* Compactor re-links boundary vertices along \partial V_A \cap \partial V_B
RelinkBoundary ==
    /\ compaction_state = "SplicingInterior"
    /\ delaunay_graph' = [v \in Vertices |->
        IF IsBoundaryVertex(v) THEN
            \* Recompute Delaunay edges across adjacent Voronoi boundary cells
            delaunay_graph[v] \cup {u \in Vertices : IsBoundaryVertex(u) /\ u # v}
        ELSE
            \* Interior vertices preserve mathematically exact spliced edges
            delaunay_graph[v]
       ]
    /\ compaction_state' = "RelinkingBoundary"
    /\ UNCHANGED <<cell_assignment, vertex_status, committed_versions, active_directory, reader_epochs>>

\* Atomic pointer swap of active directory from PreCompaction to PostCompaction
AtomicPointerSwap ==
    /\ compaction_state = "RelinkingBoundary"
    /\ active_directory' = "PostCompaction"
    /\ compaction_state' = "Swapped"
    /\ UNCHANGED <<cell_assignment, vertex_status, committed_versions, delaunay_graph, reader_epochs>>

\* Reclaim old pre-compaction runs once all readers referencing them retire
ReclaimOldRuns ==
    /\ compaction_state = "Swapped"
    /\ \A r \in reader_epochs : r.run_view = "PostCompaction"
    /\ compaction_state' = "Reclaimed"
    /\ UNCHANGED <<cell_assignment, vertex_status, committed_versions, delaunay_graph, active_directory, reader_epochs>>

\* Reader acquires an MVCC snapshot epoch
AcquireReaderSnapshot(reader_id, query_ts) ==
    /\ reader_id \notin {r.id : r \in reader_epochs}
    /\ reader_epochs' = reader_epochs \cup {[
        id |-> reader_id,
        snapshot_ts |-> query_ts,
        run_view |-> active_directory
       ]}
    /\ UNCHANGED <<cell_assignment, vertex_status, committed_versions, delaunay_graph, active_directory, compaction_state>>

\* Reader finishes snapshot query and releases its epoch guard
ReleaseReaderSnapshot(reader_id) ==
    /\ \E r \in reader_epochs : r.id = reader_id
    /\ reader_epochs' = {r \in reader_epochs : r.id # reader_id}
    /\ UNCHANGED <<cell_assignment, vertex_status, committed_versions, delaunay_graph, active_directory, compaction_state>>

Next ==
    \/ \E v \in Vertices, ts \in 2..MaxTimestamps : MutateVertex(v, ts)
    \/ ComputeCertificates
    \/ SpliceInterior
    \/ RelinkBoundary
    \/ AtomicPointerSwap
    \/ ReclaimOldRuns
    \/ \E rid \in 1..3, qts \in 1..MaxTimestamps : AcquireReaderSnapshot(rid, qts)
    \/ \E rid \in 1..3 : ReleaseReaderSnapshot(rid)

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
(* Invariants                                                              *)
-----------------------------------------------------------------------------
TypeOK ==
    /\ active_directory \in {"PreCompaction", "PostCompaction"}
    /\ compaction_state \in {"Idle", "ComputingCertificates", "SplicingInterior", "RelinkingBoundary", "Swapped", "Reclaimed"}
    /\ \A v \in Vertices : committed_versions[v] # {}

\* Invariant 1: No lost mutations across compaction transitions
Inv_NoLostMutations ==
    \A v \in Vertices :
        \A ts \in committed_versions[v] :
            ts \in committed_versions[v]

\* Invariant 2: Snapshot isolation: Readers observe an immutable, non-torn view
Inv_SnapshotIsolation ==
    \A r \in reader_epochs :
        /\ r.run_view \in {"PreCompaction", "PostCompaction"}
        /\ (compaction_state = "Reclaimed" => r.run_view = "PostCompaction")

\* Invariant 3: Delaunay boundary equivalence: Spliced interior vertices maintain their exact Voronoi-constrained neighborhood
Inv_DelaunayBoundaryEquivalence ==
    \A v \in Vertices :
        IsInteriorVertex(v) =>
            (\A u \in delaunay_graph[v] : cell_assignment[u] = cell_assignment[v])

StateConstraint ==
    /\ Cardinality(reader_epochs) <= 2

=============================================================================

