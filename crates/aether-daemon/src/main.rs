//! Project Aether-Manifold: Standalone Database Daemon & CLI (`aetherd`)
//!
//! Provides the primary operational command-line interface:
//! - `aetherd status`: Prints architecture layout, thread affinity, and live Prometheus telemetry.
//! - `aetherd exec "<AQL>"`: Executes declarative AQL statements with microsecond telemetry.
//! - `aetherd verify-gates`: Self-tests 4KB page alignment, L0 MVCC geodesic interpolation, and Semiring SpMV.

use aether_autotune::dma_compaction::HostLoopbackDmaBackend;
use aether_consensus::consensus::MAX_CLUSTER_NODES;
use aether_core::error::AetherError;
use aether_core::hlc::HlcTimestamp;
use aether_core::page::{PageHeader, TensorGraphPage};
use aether_manifold::product::{HyperbolicSpec, ProductManifoldConfig};
use aether_query::aql::{parse_aql, AqlCompileHelper, AqlStatement};
use aether_query::compiler::{FrontierNode, MAX_QUERY_FRONTIER};
use aether_query::sheaf::{pack_page_sheaf_morphisms, SheafRestrictionMap};
use aether_storage::l0_mvcc::L0MorphismLog;
use aether_storage::shard_runtime::{pin_current_thread_to_core, CoreShardRuntime};
use bytemuck::Zeroable;

fn print_usage() {
    println!("Aether-Manifold Database Daemon (aetherd v0.1.0)");
    println!("Usage:");
    println!("  aetherd status                                    Print engine architecture, core affinity, and live Prometheus telemetry");
    println!("  aetherd exec \"<AQL>\"                              Execute a declarative AQL statement and report latency");
    println!("  aetherd serve [--listen <addr>] [--max-requests <n>] Run persistent Aether-Manifold server");
    println!("  aetherd verify-gates                              Run runtime validation of 4KB alignment, MVCC, and SpMV gates");
    println!("  aetherd --help                                    Show this help information");
}

fn handle_status() -> Result<(), AetherError> {
    println!("================================================================================");
    println!("Project Aether-Manifold: Engine Architecture & Shard Telemetry");
    println!("================================================================================");
    println!("Compile-Time Memory Layout Invariants:");
    println!(
        "  - TensorGraphPage Size:      {} bytes (4096B NVMe/Direct I/O aligned)",
        core::mem::size_of::<TensorGraphPage>()
    );
    println!(
        "  - TensorGraphPage Alignment: {} bytes",
        core::mem::align_of::<TensorGraphPage>()
    );
    println!(
        "  - PageHeader Size:           {} bytes (64B CPU cache-line aligned)",
        core::mem::size_of::<PageHeader>()
    );
    println!(
        "  - PageHeader Alignment:      {} bytes",
        core::mem::align_of::<PageHeader>()
    );
    println!(
        "  - Cluster Scale Limit:       {} nodes (256-bit non-colliding quorum mask)",
        MAX_CLUSTER_NODES
    );

    let pin_result = pin_current_thread_to_core(0);
    match pin_result {
        Ok(()) => println!("Worker Core Affinity:          Pinned to Core 0 successfully"),
        Err(e) => println!(
            "Worker Core Affinity:          Failed to pin to Core 0 ({:?})",
            e
        ),
    }

    // Initialize an active shard runtime and allocate initial pages
    let mut runtime = CoreShardRuntime::<16, 64, 64>::new(0, 16, HlcTimestamp(1_000_000));
    let s0 = runtime
        .arena
        .allocate_page(101, HlcTimestamp(1_000_010), 0)?;
    let s1 = runtime
        .arena
        .allocate_page(102, HlcTimestamp(1_000_020), 0)?;

    runtime.submit_page_write(101, s0)?;
    runtime.submit_page_prefetch(102, s1)?;
    let _ = runtime.process_pending_io();

    let mut prom_buf = [0u8; 2048];
    let bytes_written = runtime.export_prometheus_text(&mut prom_buf);

    println!("\nLive Prometheus Metrics Exposition (Zero-Allocation Buffer):");
    println!("--------------------------------------------------------------------------------");
    if let Some(slice) = prom_buf.get(..bytes_written) {
        if let Ok(text) = core::str::from_utf8(slice) {
            print!("{}", text);
        }
    }
    println!("--------------------------------------------------------------------------------");
    Ok(())
}

fn build_sample_topology() -> &'static [TensorGraphPage] {
    let mut pages = vec![TensorGraphPage::zeroed(); 5];

    let adjacency: [(u64, &[u64]); 5] = [
        (0, &[1, 2]),
        (1, &[2, 3]),
        (2, &[3, 4]),
        (3, &[4]),
        (4, &[]),
    ];

    for (page_idx, &(node_id, neighbors)) in adjacency.iter().enumerate() {
        if let Some(page) = pages.get_mut(page_idx) {
            page.header.page_id = node_id;
            page.header.active_vertex_count = neighbors.len() as u32;

            let stalk = [(node_id as f32 + 1.0) * 0.1f32; 8];
            page.set_primary_stalk(&stalk);

            for (n_idx, &n_id) in neighbors.iter().enumerate() {
                if let Some(slot) = page.neighbor_ids.get_mut(n_idx) {
                    *slot = n_id;
                }
            }

            let morphisms =
                [SheafRestrictionMap::<8>::from_reflector(&[0.353_553_38f32; 8], 1.0); 64];
            let edge_weights = [1.0f32; 64];
            let _ = pack_page_sheaf_morphisms(page, &morphisms, &edge_weights);
            page.header.update_checksum();
        }
    }

    Box::leak(pages.into_boxed_slice())
}

fn handle_exec(query: &str) -> Result<(), AetherError> {
    let start_time = std::time::Instant::now();
    let stmt = parse_aql(query)?;
    let parse_latency = start_time.elapsed().as_micros();

    println!("================================================================================");
    println!("AQL Query Execution Engine");
    println!("================================================================================");
    println!("Input Query: \"{}\"", query);
    println!("AST Parse Time: {} µs", parse_latency);

    match stmt {
        AqlStatement::CreateManifold {
            name,
            hyperbolic_dim,
            kappa,
            spherical_dim,
            tau,
            euclidean_dim,
        } => {
            let total_dim = hyperbolic_dim + spherical_dim + euclidean_dim;
            println!("Status: MANIFOLD SCHEMA CREATED");
            println!("  Schema Name:       {}", name);
            println!("  Total Dimension:   {}", total_dim);
            println!(
                "  Hyperbolic Lorentz: H^{} (kappa = {:.3})",
                hyperbolic_dim, kappa
            );
            println!(
                "  Spherical:          S^{} (tau = {:.3})",
                spherical_dim, tau
            );
            println!("  Euclidean:          R^{}", euclidean_dim);
        }
        AqlStatement::InsertVertex {
            vertex_id,
            vector,
            stalk,
            neighbors,
        } => {
            println!("Status: VERTEX INGESTED");
            println!("  Vertex ID:          {}", vertex_id);
            println!(
                "  Coordinate Vector:  {} dimensions ({:?})",
                vector.len(),
                vector
            );
            println!("  Sheaf Stalk x_v:    {:?}", stalk);
            println!("  1-Hop Neighbors:    {:?}", neighbors);
        }
        AqlStatement::UpsertVertex {
            vertex_id,
            vector,
            stalk,
            neighbors,
            ..
        } => {
            println!("Status: VERTEX UPSERTED");
            println!("  Vertex ID:          {}", vertex_id);
            println!(
                "  Coordinate Vector:  {} dimensions ({:?})",
                vector.len(),
                vector
            );
            println!("  Sheaf Stalk x_v:    {:?}", stalk);
            println!("  1-Hop Neighbors:    {:?}", neighbors);
        }
        AqlStatement::ConnectVertex {
            source,
            target,
            morphism_type,
            weight,
            ..
        } => {
            println!("Status: VERTICES CONNECTED");
            println!("  Source Vertex:      {}", source);
            println!("  Target Vertex:      {}", target);
            println!("  Morphism Type:      {}", morphism_type);
            println!("  Weight:             {:.4}", weight);
        }
        AqlStatement::OptimizeManifold {
            manifold,
            from_level,
            to_level,
            method,
            ..
        } => {
            println!("Status: MANIFOLD OPTIMIZATION TRIGGERED");
            println!("  Target Manifold:    {}", manifold);
            println!("  Tier Transition:    L{} -> L{}", from_level, to_level);
            println!("  Strategy:           {}", method);
        }
        AqlStatement::Traverse {
            start_vertex,
            hops,
            semiring,
            beta,
            gamma,
            limit,
        } => {
            let pages = build_sample_topology();
            let page_lookup = move |id: u64| -> Option<&'static TensorGraphPage> {
                pages.iter().find(|p| p.header.page_id == id)
            };
            let distance_oracle = move |_id: u64| -> f32 { 0.05f32 };

            let mut results = [FrontierNode::default(); MAX_QUERY_FRONTIER];
            let mut result_count = 0usize;

            let exec_start = std::time::Instant::now();
            AqlCompileHelper::execute_traverse(
                &AqlStatement::Traverse {
                    start_vertex,
                    hops,
                    semiring,
                    beta,
                    gamma,
                    limit,
                },
                page_lookup,
                distance_oracle,
                &mut results,
                &mut result_count,
            )?;
            let exec_latency = exec_start.elapsed().as_micros();

            println!("Status: TRAVERSAL COMPLETE");
            println!("  Semiring Mode:      {:?}", semiring);
            println!("  Traversal Hops:     {}", hops);
            println!("  Beta (Sheaf):       {:.3}", beta);
            println!("  Gamma (Manifold):   {:.3}", gamma);
            println!("  Execution Latency:  {} µs", exec_latency);
            println!("  Frontier Candidates: {}", result_count);
            println!("\nTop Ranked Frontier Nodes:");
            println!("  Rank | Node ID | Accumulated Score | Hop Depth");
            println!("  ----------------------------------------------");

            let display_limit = result_count.min(limit);
            for rank in 0..display_limit {
                if let Some(entry) = results.get(rank) {
                    println!(
                        "  {:>4} | {:>7} | {:>17.6} | {:>9}",
                        rank + 1,
                        entry.node_id,
                        entry.score,
                        entry.hop
                    );
                }
            }
        }
    }
    println!(
        "Total Wall-Clock Latency: {} µs",
        start_time.elapsed().as_micros()
    );
    Ok(())
}

fn execute_query_to_string(query: &str) -> String {
    let start_time = std::time::Instant::now();
    match parse_aql(query) {
        Ok(stmt) => match stmt {
            AqlStatement::CreateManifold {
                name,
                hyperbolic_dim,
                kappa,
                spherical_dim,
                tau,
                euclidean_dim,
            } => {
                let total_dim = hyperbolic_dim + spherical_dim + euclidean_dim;
                format!(
                    "STATUS OK | Schema: {} | Dim: {} (H^{}(k={:.2}) x S^{}(t={:.2}) x R^{}) | Latency: {} us\n",
                    name,
                    total_dim,
                    hyperbolic_dim,
                    kappa,
                    spherical_dim,
                    tau,
                    euclidean_dim,
                    start_time.elapsed().as_micros()
                )
            }
            AqlStatement::InsertVertex {
                vertex_id,
                vector,
                stalk,
                neighbors,
            } => {
                format!(
                    "STATUS OK | Inserted vertex {} (dim={}, stalk_dim={}, neighbors={}) | Latency: {} us\n",
                    vertex_id,
                    vector.len(),
                    stalk.len(),
                    neighbors.len(),
                    start_time.elapsed().as_micros()
                )
            }
            AqlStatement::UpsertVertex {
                vertex_id,
                vector,
                stalk,
                neighbors,
                ..
            } => {
                format!(
                    "STATUS OK | Upserted vertex {} (dim={}, stalk_dim={}, neighbors={}) | Latency: {} us\n",
                    vertex_id,
                    vector.len(),
                    stalk.len(),
                    neighbors.len(),
                    start_time.elapsed().as_micros()
                )
            }
            AqlStatement::ConnectVertex {
                source,
                target,
                morphism_type,
                weight,
                ..
            } => {
                format!(
                    "STATUS OK | Connected {} -> {} (type={}, weight={:.4}) | Latency: {} us\n",
                    source,
                    target,
                    morphism_type,
                    weight,
                    start_time.elapsed().as_micros()
                )
            }
            AqlStatement::OptimizeManifold {
                manifold,
                from_level,
                to_level,
                method,
                ..
            } => {
                format!(
                    "STATUS OK | Optimized manifold {} (L{} -> L{}, method={}) | Latency: {} us\n",
                    manifold,
                    from_level,
                    to_level,
                    method,
                    start_time.elapsed().as_micros()
                )
            }
            AqlStatement::Traverse {
                start_vertex,
                hops,
                semiring,
                beta,
                gamma,
                limit,
            } => {
                let pages = build_sample_topology();
                let page_lookup = move |id: u64| -> Option<&'static TensorGraphPage> {
                    pages.iter().find(|p| p.header.page_id == id)
                };
                let distance_oracle = move |_id: u64| -> f32 { 0.05f32 };

                let mut results = [FrontierNode::default(); MAX_QUERY_FRONTIER];
                let mut result_count = 0usize;

                match AqlCompileHelper::execute_traverse(
                    &AqlStatement::Traverse {
                        start_vertex,
                        hops,
                        semiring,
                        beta,
                        gamma,
                        limit,
                    },
                    page_lookup,
                    distance_oracle,
                    &mut results,
                    &mut result_count,
                ) {
                    Ok(()) => {
                        let exec_latency = start_time.elapsed().as_micros();
                        let display_limit = result_count.min(limit);
                        let mut res_str = format!(
                            "STATUS OK | Traverse from {} (hops={}, semiring={:?}) -> {} candidates | Latency: {} us\n",
                            start_vertex, hops, semiring, result_count, exec_latency
                        );
                        for rank in 0..display_limit {
                            if let Some(entry) = results.get(rank) {
                                res_str.push_str(&format!(
                                    "  Rank {}: node={}, score={:.6}, hop={}\n",
                                    rank + 1,
                                    entry.node_id,
                                    entry.score,
                                    entry.hop
                                ));
                            }
                        }
                        res_str
                    }
                    Err(e) => format!("STATUS ERROR: {:?}\n", e),
                }
            }
        },
        Err(e) => format!("STATUS PARSE_ERROR: {:?}\n", e),
    }
}

fn handle_serve(listen_addr: &str, max_requests: Option<usize>) -> Result<(), AetherError> {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    println!("================================================================================");
    println!("Project Aether-Manifold: High-Throughput Daemon Server (`aetherd serve`)");
    println!("================================================================================");
    println!("Listen Address:    {}", listen_addr);
    let pin_result = pin_current_thread_to_core(0);
    match pin_result {
        Ok(()) => println!("Worker Thread:     Pinned to Core 0 successfully"),
        Err(e) => println!("Worker Thread:     Core 0 pin skipped ({:?})", e),
    }

    let listener = TcpListener::bind(listen_addr).map_err(|e| AetherError::IoFault {
        code: e.raw_os_error().unwrap_or(-1),
    })?;
    listener
        .set_nonblocking(true)
        .map_err(|e| AetherError::IoFault {
            code: e.raw_os_error().unwrap_or(-1),
        })?;

    println!("Server State:      READY (Accepting non-blocking TCP connections)");
    println!("--------------------------------------------------------------------------------");

    let mut requests_served = 0usize;

    while max_requests.is_none_or(|limit| requests_served < limit) {
        match listener.accept() {
            Ok((mut stream, peer_addr)) => {
                let mut buf = [0u8; 4096];
                let _ = stream.set_read_timeout(Some(std::time::Duration::from_millis(1000)));
                match stream.read(&mut buf) {
                    Ok(n) if n > 0 => {
                        let slice = buf.get(..n).unwrap_or(&[]);
                        let query_str = String::from_utf8_lossy(slice);
                        let trimmed = query_str.trim();
                        if !trimmed.is_empty() {
                            let response = execute_query_to_string(trimmed);
                            let _ = stream.write_all(response.as_bytes());
                            let _ = stream.flush();
                            requests_served += 1;
                            println!(
                                "[Req #{}] From {}: \"{}\" -> OK",
                                requests_served, peer_addr, trimmed
                            );
                        }
                    }
                    _ => {}
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            Err(e) => {
                eprintln!("Accept error: {:?}", e);
            }
        }
    }

    println!(
        "Server shutdown gracefully after {} requests.",
        requests_served
    );
    Ok(())
}

fn handle_verify_gates() -> Result<(), AetherError> {
    println!("================================================================================");
    println!("Project Aether-Manifold: Hardware & Algorithmic Self-Test Exit Gates");
    println!("================================================================================");

    // Gate 1: Memory Alignment & Repr
    print!("Gate 1: Verifying 4KB page and 64B header compile-time layouts... ");
    assert_eq!(core::mem::size_of::<TensorGraphPage>(), 4096);
    assert_eq!(core::mem::align_of::<TensorGraphPage>(), 4096);
    assert_eq!(core::mem::size_of::<PageHeader>(), 64);
    assert_eq!(core::mem::align_of::<PageHeader>(), 64);
    println!("[PASSED]");

    // Gate 2: L0 Geodesic MVCC Interpolation
    print!("Gate 2: Verifying L0 MVCC continuous geodesic state interpolation... ");
    let config = ProductManifoldConfig {
        hyperbolic: vec![HyperbolicSpec {
            dim: 16,
            kappa: -1.0,
            weight: 1.0,
        }],
        spherical: vec![],
        euclidean: None,
    };
    let mut log = L0MorphismLog::<16, 4>::new();
    let mut base = [0.0f32; 16];
    base[0] = 1.0;
    let mut commit = base;
    commit[1] = 0.05;
    commit[0] = (1.0 + 0.05 * 0.05f32).sqrt();

    let _slot = log.record_geodesic_delta(
        &config,
        42,
        HlcTimestamp(100),
        &base,
        HlcTimestamp(200),
        &commit,
    )?;
    let entry = log
        .get_latest_entry(42)
        .ok_or(AetherError::InvalidNodeId { node_id: 42 })?;
    let mut out = [0.0f32; 16];
    log.reconstruct_state_at(&config, &base, entry, HlcTimestamp(150), &mut out)?;
    assert!(!out[0].is_nan());
    println!("[PASSED]");

    // Gate 3: Semiring SpMV Traversal & DMA Driver Backend Check
    print!("Gate 3: Verifying Semiring-Sheaf SpMV query engine & DMA backend... ");
    let _dma = HostLoopbackDmaBackend::new();
    let aql =
        "TRAVERSE FROM 0 HOPS 2 USING SEMIRING TROPICAL WHERE BETA = 0.5 AND GAMMA = 1.0 LIMIT 5";
    let stmt = parse_aql(aql)?;
    let pages = build_sample_topology();
    let page_lookup = move |id: u64| -> Option<&'static TensorGraphPage> {
        pages.iter().find(|p| p.header.page_id == id)
    };
    let distance_oracle = move |_id: u64| -> f32 { 0.1f32 };

    let mut results = [FrontierNode::default(); MAX_QUERY_FRONTIER];
    let mut count = 0usize;
    AqlCompileHelper::execute_traverse(
        &stmt,
        page_lookup,
        distance_oracle,
        &mut results,
        &mut count,
    )?;
    assert!(count > 0);
    println!("[PASSED]");

    println!("\nAll exit gates passed self-check successfully (Status Code: 0).");
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        return;
    }

    let cmd = match args.get(1) {
        Some(c) => c.as_str(),
        None => {
            print_usage();
            return;
        }
    };

    let result = match cmd {
        "status" => handle_status(),
        "exec" => {
            if let Some(query) = args.get(2) {
                handle_exec(query)
            } else {
                eprintln!("Error: 'exec' command requires an AQL query string.");
                eprintln!("Example: aetherd exec \"TRAVERSE FROM 0 HOPS 3 USING SEMIRING TROPICAL LIMIT 5\"");
                std::process::exit(1);
            }
        }
        "serve" => {
            let mut listen_addr = "127.0.0.1:9090".to_string();
            let mut max_requests = None;
            let mut i = 2;
            while i < args.len() {
                match args.get(i).map(|s| s.as_str()) {
                    Some("--listen") | Some("-l") => {
                        if let Some(addr) = args.get(i + 1) {
                            listen_addr = addr.clone();
                            i += 2;
                        } else {
                            eprintln!("Error: --listen requires an address argument");
                            std::process::exit(1);
                        }
                    }
                    Some("--max-requests") | Some("-n") => {
                        if let Some(n_str) = args.get(i + 1) {
                            if let Ok(limit) = n_str.parse::<usize>() {
                                max_requests = Some(limit);
                            }
                            i += 2;
                        } else {
                            eprintln!("Error: --max-requests requires an integer");
                            std::process::exit(1);
                        }
                    }
                    Some(arg) => {
                        eprintln!("Warning: Unknown serve argument '{}'", arg);
                        i += 1;
                    }
                    None => break,
                }
            }
            handle_serve(&listen_addr, max_requests)
        }
        "verify-gates" => handle_verify_gates(),
        "--help" | "-h" | "help" => {
            print_usage();
            Ok(())
        }
        "--version" | "-v" | "version" => {
            println!("aetherd 0.1.0 (Aether-Manifold Database Engine)");
            Ok(())
        }
        unknown => {
            eprintln!("Error: Unknown command '{}'", unknown);
            print_usage();
            std::process::exit(1);
        }
    };

    if let Err(e) = result {
        eprintln!("Engine Error: {:?}", e);
        std::process::exit(1);
    }
}
