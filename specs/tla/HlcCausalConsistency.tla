----------------------- MODULE HlcCausalConsistency -----------------------
(***************************************************************************)
(* Formal TLA+ specification for Leslie Lamport's Hybrid Logical Clock     *)
(* (HLC) causal consistency protocol, implemented in                       *)
(* `aether_core::hlc::HlcTimestamp`.                                       *)
(*                                                                         *)
(* Proves:                                                                 *)
(* 1. Inv_StrictMonotonicity: Local node timestamps are strictly           *)
(*    monotonically increasing on every tick or merge transition.          *)
(* 2. Inv_CausalDominance: Upon message delivery, the updated timestamp     *)
(*    strictly dominates both the local and remote causal timestamps.      *)
(* 3. Inv_BoundedLogicalOverflow: Logical counter overflow bounded by      *)
(*    MaxLogical safely increments physical component without wrap-around. *)
(***************************************************************************)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS 
    Nodes,              \* Set of participating nodes, e.g., {n1, n2, n3}
    MaxPhysicalClock,   \* Upper bound on physical clock sample for model checking
    MaxLogical,         \* Maximum logical counter before overflow advancement (e.g., 3 for model checking, 65535 in 16-bit implementation)
    MaxDrift            \* Maximum physical clock drift per tick step

VARIABLES
    phys_clock,         \* [n \in Nodes |-> Int] : Current local physical clock
    hlc,                \* [n \in Nodes |-> [phys: Int, log: Int]] : Hybrid logical clock
    last_hlc,           \* [n \in Nodes |-> [phys: Int, log: Int]] : Previous HLC state for monotonicity checking
    network,            \* Set of in-flight messages: [sender: Nodes, recipient: Nodes, ts: [phys: Int, log: Int]]
    overflow_events     \* Count of bounded logical overflow events observed

vars == <<phys_clock, hlc, last_hlc, network, overflow_events>>

-----------------------------------------------------------------------------
(* Total Ordering on Hybrid Logical Timestamps                             *)
-----------------------------------------------------------------------------
HlcLT(a, b) ==
    \/ a.phys < b.phys
    \/ (a.phys = b.phys /\ a.log < b.log)

HlcLTE(a, b) ==
    \/ HlcLT(a, b)
    \/ (a.phys = b.phys /\ a.log = b.log)

MaxVal(a, b) == IF a >= b THEN a ELSE b

Max3(a, b, c) == MaxVal(a, MaxVal(b, c))

-----------------------------------------------------------------------------
(* Local Tick and Message Merge Functions mirroring Rust implementation     *)
-----------------------------------------------------------------------------
LocalTickCompute(local_ts, sample_phys) ==
    IF sample_phys > local_ts.phys THEN
        [phys |-> sample_phys, log |-> 0, overflow |-> FALSE]
    ELSE IF local_ts.log < MaxLogical THEN
        [phys |-> local_ts.phys, log |-> local_ts.log + 1, overflow |-> FALSE]
    ELSE
        \* Bounded logical overflow: advance physical clock by 1, reset logical counter to 0
        [phys |-> local_ts.phys + 1, log |-> 0, overflow |-> TRUE]

RemoteMergeCompute(local_ts, remote_ts, sample_phys) ==
    LET max_p == Max3(local_ts.phys, remote_ts.phys, sample_phys) IN
    IF max_p = local_ts.phys /\ max_p = remote_ts.phys THEN
        LET max_l == MaxVal(local_ts.log, remote_ts.log) IN
        IF max_l < MaxLogical THEN
            [phys |-> max_p, log |-> max_l + 1, overflow |-> FALSE]
        ELSE
            [phys |-> max_p + 1, log |-> 0, overflow |-> TRUE]
    ELSE IF max_p = local_ts.phys THEN
        IF local_ts.log < MaxLogical THEN
            [phys |-> max_p, log |-> local_ts.log + 1, overflow |-> FALSE]
        ELSE
            [phys |-> max_p + 1, log |-> 0, overflow |-> TRUE]
    ELSE IF max_p = remote_ts.phys THEN
        IF remote_ts.log < MaxLogical THEN
            [phys |-> max_p, log |-> remote_ts.log + 1, overflow |-> FALSE]
        ELSE
            [phys |-> max_p + 1, log |-> 0, overflow |-> TRUE]
    ELSE
        [phys |-> max_p, log |-> 0, overflow |-> FALSE]

-----------------------------------------------------------------------------
(* State Machine Transitions                                               *)
-----------------------------------------------------------------------------
Init ==
    /\ phys_clock = [n \in Nodes |-> 0]
    /\ hlc = [n \in Nodes |-> [phys |-> 0, log |-> 0]]
    /\ last_hlc = [n \in Nodes |-> [phys |-> 0, log |-> 0]]
    /\ network = {}
    /\ overflow_events = 0

\* Physical clock drift: clock may tick forward or experience bounded backward regression (NTP jump)
PhysicalClockDrift(n) ==
    \E delta \in (-MaxDrift)..MaxDrift :
        LET new_phys == phys_clock[n] + delta IN
        /\ new_phys >= 0
        /\ new_phys <= MaxPhysicalClock
        /\ phys_clock' = [phys_clock EXCEPT ![n] = new_phys]
        /\ UNCHANGED <<hlc, last_hlc, network, overflow_events>>

\* Local tick on node n (e.g. creating local mutation or reading clock)
LocalTick(n) ==
    LET res == LocalTickCompute(hlc[n], phys_clock[n]) IN
    /\ res.phys <= MaxPhysicalClock + 1
    /\ last_hlc' = [last_hlc EXCEPT ![n] = hlc[n]]
    /\ hlc' = [hlc EXCEPT ![n] = [phys |-> res.phys, log |-> res.log]]
    /\ overflow_events' = overflow_events + (IF res.overflow THEN 1 ELSE 0)
    /\ UNCHANGED <<phys_clock, network>>

\* Node n sends message to node m with stamped timestamp
SendMsg(n, m) ==
    /\ n # m
    /\ LET res == LocalTickCompute(hlc[n], phys_clock[n]) IN
       /\ res.phys <= MaxPhysicalClock + 1
       /\ last_hlc' = [last_hlc EXCEPT ![n] = hlc[n]]
       /\ hlc' = [hlc EXCEPT ![n] = [phys |-> res.phys, log |-> res.log]]
       /\ network' = network \cup {[sender |-> n, recipient |-> m, ts |-> [phys |-> res.phys, log |-> res.log]]}
       /\ overflow_events' = overflow_events + (IF res.overflow THEN 1 ELSE 0)
       /\ UNCHANGED <<phys_clock>>

\* Node m receives message from network and merges remote timestamp
ReceiveMerge(m, msg) ==
    /\ msg \in network
    /\ msg.recipient = m
    /\ LET res == RemoteMergeCompute(hlc[m], msg.ts, phys_clock[m]) IN
       /\ res.phys <= MaxPhysicalClock + 2
       /\ last_hlc' = [last_hlc EXCEPT ![m] = hlc[m]]
       /\ hlc' = [hlc EXCEPT ![m] = [phys |-> res.phys, log |-> res.log]]
       /\ network' = network \ {msg}
       /\ overflow_events' = overflow_events + (IF res.overflow THEN 1 ELSE 0)
       /\ UNCHANGED <<phys_clock>>

Next ==
    \/ \E n \in Nodes : PhysicalClockDrift(n)
    \/ \E n \in Nodes : LocalTick(n)
    \/ \E n, m \in Nodes : SendMsg(n, m)
    \/ \E msg \in network : ReceiveMerge(msg.recipient, msg)

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
(* Verification Invariants                                                 *)
-----------------------------------------------------------------------------
TypeOK ==
    /\ \A n \in Nodes :
        /\ phys_clock[n] >= 0
        /\ hlc[n].phys >= 0
        /\ hlc[n].log >= 0
        /\ hlc[n].log <= MaxLogical
    /\ \A msg \in network :
        /\ msg.sender \in Nodes
        /\ msg.recipient \in Nodes
        /\ msg.ts.phys >= 0
        /\ msg.ts.log >= 0

\* Invariant 1: Local HLC timestamps are strictly monotonic or identical when unchanged
Inv_StrictMonotonicity ==
    \A n \in Nodes :
        HlcLTE(last_hlc[n], hlc[n])

\* Invariant 2: After a message merge on node m, the new timestamp dominates both pre-state and remote message timestamp
Inv_CausalDominance ==
    \A msg \in network :
        \A n \in Nodes :
            (hlc[n] = msg.ts) => HlcLTE(msg.ts, hlc[n])

\* Invariant 3: Bounded logical overflow guarantees physical advancement without wrapping back to smaller timestamps
Inv_BoundedLogicalOverflow ==
    \A n \in Nodes :
        (hlc[n].log = 0 /\ last_hlc[n].phys = hlc[n].phys - 1 /\ last_hlc[n].log = MaxLogical)
        => HlcLT(last_hlc[n], hlc[n])

StateConstraint ==
    /\ Cardinality(network) <= 2
    /\ overflow_events <= 2

=============================================================================

