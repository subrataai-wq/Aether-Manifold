//! Project Aether-Manifold: Declarative Aether Query Language (AQL)
//!
//! Provides an AST, lexer, and parser for declarative non-Euclidean database operations:
//! - `CREATE MANIFOLD`: Configures composite Lorentz, Spherical, and Euclidean components.
//! - `INSERT VERTEX`: Ingests multi-manifold vectors, sheaf stalks, and 1-hop topology.
//! - `TRAVERSE`: Compiles declarative graph queries into Generalized Semiring SpMV.
//!
//! Zero external dependencies; zero hot-path allocations during plan execution (Rule 3.1).

use aether_core::error::AetherError;
use aether_core::page::TensorGraphPage;

use crate::compiler::{FrontierNode, QueryPlan, SemiringSpMvEngine, MAX_QUERY_FRONTIER};
use crate::semiring::SemiringMode;

/// Runtime semiring operational kind for AQL query planning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemiringKind {
    /// Tropical semiring $(\min, +)$ for geodesic shortest paths.
    Tropical,
    /// Diffusion semiring $(+, \times)$ for Sheaf heat diffusion.
    Diffusion,
    /// Bottleneck semiring $(\max, \min)$ for capacity routing.
    Bottleneck,
}

impl From<SemiringKind> for SemiringMode {
    fn from(kind: SemiringKind) -> Self {
        match kind {
            SemiringKind::Tropical => SemiringMode::Tropical,
            SemiringKind::Diffusion => SemiringMode::Diffusion,
            SemiringKind::Bottleneck => SemiringMode::Bottleneck,
        }
    }
}

impl From<SemiringMode> for SemiringKind {
    fn from(mode: SemiringMode) -> Self {
        match mode {
            SemiringMode::Tropical => SemiringKind::Tropical,
            SemiringMode::Diffusion => SemiringKind::Diffusion,
            SemiringMode::Bottleneck => SemiringKind::Bottleneck,
        }
    }
}

/// Abstract Syntax Tree (AST) for declarative Aether Query Language (AQL) statements.
#[derive(Clone, Debug, PartialEq)]
pub enum AqlStatement {
    /// Create a non-Euclidean product manifold schema.
    CreateManifold {
        /// Unique schema identifier.
        name: String,
        /// Hyperbolic Lorentz component dimension.
        hyperbolic_dim: usize,
        /// Negative sectional curvature ($\kappa < 0$).
        kappa: f32,
        /// Spherical component dimension.
        spherical_dim: usize,
        /// Spherical radius parameter ($\tau > 0$).
        tau: f32,
        /// Euclidean flat component dimension.
        euclidean_dim: usize,
    },
    /// Ingest a vertex with its manifold vector, primary stalk, and 1-hop topology.
    InsertVertex {
        /// Target vertex identifier.
        vertex_id: u64,
        /// Multi-manifold coordinate vector $\hat{x}_v$.
        vector: Vec<f32>,
        /// Primary cellular sheaf stalk $x_v \in \mathbb{R}^8$.
        stalk: [f32; 8],
        /// Neighbor node identifiers $\mathcal{N}(v)$.
        neighbors: Vec<u64>,
    },
    /// Execute a hybrid semiring-sheaf multi-hop graph traversal.
    Traverse {
        /// Seed starting vertex.
        start_vertex: u64,
        /// Traversal depth (number of SpMV iterations).
        hops: usize,
        /// Semiring algebraic framework (Tropical, Diffusion, Bottleneck).
        semiring: SemiringKind,
        /// Sheaf Dirichlet energy sensitivity weight $\beta \ge 0$.
        beta: f32,
        /// Riemannian geodesic distance sensitivity weight $\gamma \ge 0$.
        gamma: f32,
        /// Result limit (Top-K).
        limit: usize,
    },
    /// Upsert or insert a vertex with multi-manifold vector, primary stalk, and 1-hop topology.
    UpsertVertex {
        /// Target vertex identifier.
        vertex_id: u64,
        /// Target manifold identifier (optional).
        manifold: Option<String>,
        /// Multi-manifold coordinate vector $\hat{x}_v$.
        vector: Vec<f32>,
        /// Primary cellular sheaf stalk $x_v \in \mathbb{R}^8$.
        stalk: [f32; 8],
        /// Neighbor node identifiers $\mathcal{N}(v)$.
        neighbors: Vec<u64>,
    },
    /// Connect two vertices with a cellular sheaf morphism and scalar weight.
    ConnectVertex {
        /// Source vertex identifier.
        source: u64,
        /// Destination vertex identifier.
        target: u64,
        /// Associated manifold name (optional).
        manifold: Option<String>,
        /// Morphism specification (e.g. "HOUSEHOLDER" or "IDENTITY").
        morphism_type: String,
        /// Scalar affinity weight.
        weight: f32,
    },
    /// Trigger an autonomous tier compaction or Ricci curvature optimization.
    OptimizeManifold {
        /// Target manifold identifier.
        manifold: String,
        /// Source tier level (e.g. 1 for L1).
        from_level: u8,
        /// Target tier level (e.g. 2 for L2).
        to_level: u8,
        /// Compaction strategy: e.g. "VORONOI_BOUNDARY_CERTIFICATES" or "RICCI_FLOW".
        method: String,
        /// Whether to return telemetry metrics upon completion.
        return_telemetry: bool,
    },
}

#[derive(Debug, PartialEq, Clone)]
enum Token<'a> {
    Word(&'a str),
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Equals,
}

fn tokenize(input: &str) -> Vec<Token<'_>> {
    let mut tokens = Vec::new();
    let bytes = input.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        let b = match bytes.get(i) {
            Some(&val) => val,
            None => break,
        };

        match b {
            b' ' | b'\t' | b'\r' | b'\n' => {
                i += 1;
            }
            b'(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            b')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            b'[' => {
                tokens.push(Token::LBracket);
                i += 1;
            }
            b']' => {
                tokens.push(Token::RBracket);
                i += 1;
            }
            b',' => {
                tokens.push(Token::Comma);
                i += 1;
            }
            b'=' => {
                tokens.push(Token::Equals);
                i += 1;
            }
            _ => {
                let start = i;
                while i < len {
                    let next = match bytes.get(i) {
                        Some(&val) => val,
                        None => break,
                    };
                    if matches!(
                        next,
                        b' ' | b'\t' | b'\r' | b'\n' | b'(' | b')' | b'[' | b']' | b',' | b'='
                    ) {
                        break;
                    }
                    i += 1;
                }
                if let Some(word) = input.get(start..i) {
                    tokens.push(Token::Word(word));
                }
            }
        }
    }
    tokens
}

struct Parser<'a> {
    tokens: Vec<Token<'a>>,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(tokens: Vec<Token<'a>>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token<'a>> {
        self.tokens.get(self.pos)
    }

    fn next(&mut self) -> Option<Token<'a>> {
        let tok = self.tokens.get(self.pos).cloned();
        if tok.is_some() {
            self.pos += 1;
        }
        tok
    }

    fn expect_word(&mut self, expected: &str) -> Result<(), AetherError> {
        match self.next() {
            Some(Token::Word(w)) if w.eq_ignore_ascii_case(expected) => Ok(()),
            _ => Err(AetherError::QueryParseError { code: 10 }),
        }
    }

    fn expect_lparen(&mut self) -> Result<(), AetherError> {
        match self.next() {
            Some(Token::LParen) => Ok(()),
            _ => Err(AetherError::QueryParseError { code: 11 }),
        }
    }

    fn expect_lbracket(&mut self) -> Result<(), AetherError> {
        match self.next() {
            Some(Token::LBracket) => Ok(()),
            _ => Err(AetherError::QueryParseError { code: 13 }),
        }
    }

    fn parse_f32_list(&mut self) -> Result<Vec<f32>, AetherError> {
        self.expect_lbracket()?;
        let mut list = Vec::new();
        while let Some(tok) = self.peek() {
            if matches!(tok, Token::RBracket) {
                self.next();
                return Ok(list);
            }
            if matches!(tok, Token::Comma) {
                self.next();
                continue;
            }
            match self.next() {
                Some(Token::Word(w)) => {
                    let val = w
                        .parse::<f32>()
                        .map_err(|_| AetherError::QueryParseError { code: 20 })?;
                    list.push(val);
                }
                _ => return Err(AetherError::QueryParseError { code: 21 }),
            }
        }
        Err(AetherError::QueryParseError { code: 22 })
    }

    fn parse_u64_list(&mut self) -> Result<Vec<u64>, AetherError> {
        self.expect_lbracket()?;
        let mut list = Vec::new();
        while let Some(tok) = self.peek() {
            if matches!(tok, Token::RBracket) {
                self.next();
                return Ok(list);
            }
            if matches!(tok, Token::Comma) {
                self.next();
                continue;
            }
            match self.next() {
                Some(Token::Word(w)) => {
                    let val = w
                        .parse::<u64>()
                        .map_err(|_| AetherError::QueryParseError { code: 23 })?;
                    list.push(val);
                }
                _ => return Err(AetherError::QueryParseError { code: 24 }),
            }
        }
        Err(AetherError::QueryParseError { code: 25 })
    }

    fn parse_create_manifold(&mut self) -> Result<AqlStatement, AetherError> {
        self.expect_word("MANIFOLD")?;
        let name = match self.next() {
            Some(Token::Word(w)) => w.to_string(),
            _ => return Err(AetherError::QueryParseError { code: 30 }),
        };

        self.expect_lparen()?;

        let mut hyperbolic_dim = 0usize;
        let mut kappa = -1.0f32;
        let mut spherical_dim = 0usize;
        let mut tau = 1.0f32;
        let mut euclidean_dim = 0usize;

        while let Some(tok) = self.peek() {
            if matches!(tok, Token::RParen) {
                self.next();
                return Ok(AqlStatement::CreateManifold {
                    name,
                    hyperbolic_dim,
                    kappa,
                    spherical_dim,
                    tau,
                    euclidean_dim,
                });
            }
            if matches!(tok, Token::Comma) {
                self.next();
                continue;
            }

            match self.next() {
                Some(Token::Word(w)) if w.eq_ignore_ascii_case("HYPERBOLIC") => {
                    self.expect_lparen()?;
                    while let Some(param_tok) = self.peek() {
                        if matches!(param_tok, Token::RParen) {
                            self.next();
                            break;
                        }
                        if matches!(param_tok, Token::Comma) {
                            self.next();
                            continue;
                        }
                        let key = match self.next() {
                            Some(Token::Word(k)) => k,
                            _ => return Err(AetherError::QueryParseError { code: 31 }),
                        };
                        match self.next() {
                            Some(Token::Equals) => {}
                            _ => return Err(AetherError::QueryParseError { code: 32 }),
                        }
                        let val = match self.next() {
                            Some(Token::Word(v)) => v,
                            _ => return Err(AetherError::QueryParseError { code: 33 }),
                        };
                        if key.eq_ignore_ascii_case("dim") {
                            hyperbolic_dim = val
                                .parse::<usize>()
                                .map_err(|_| AetherError::QueryParseError { code: 34 })?;
                        } else if key.eq_ignore_ascii_case("kappa") {
                            kappa = val
                                .parse::<f32>()
                                .map_err(|_| AetherError::QueryParseError { code: 35 })?;
                        }
                    }
                }
                Some(Token::Word(w)) if w.eq_ignore_ascii_case("SPHERICAL") => {
                    self.expect_lparen()?;
                    while let Some(param_tok) = self.peek() {
                        if matches!(param_tok, Token::RParen) {
                            self.next();
                            break;
                        }
                        if matches!(param_tok, Token::Comma) {
                            self.next();
                            continue;
                        }
                        let key = match self.next() {
                            Some(Token::Word(k)) => k,
                            _ => return Err(AetherError::QueryParseError { code: 36 }),
                        };
                        match self.next() {
                            Some(Token::Equals) => {}
                            _ => return Err(AetherError::QueryParseError { code: 37 }),
                        }
                        let val = match self.next() {
                            Some(Token::Word(v)) => v,
                            _ => return Err(AetherError::QueryParseError { code: 38 }),
                        };
                        if key.eq_ignore_ascii_case("dim") {
                            spherical_dim = val
                                .parse::<usize>()
                                .map_err(|_| AetherError::QueryParseError { code: 39 })?;
                        } else if key.eq_ignore_ascii_case("tau") {
                            tau = val
                                .parse::<f32>()
                                .map_err(|_| AetherError::QueryParseError { code: 40 })?;
                        }
                    }
                }
                Some(Token::Word(w)) if w.eq_ignore_ascii_case("EUCLIDEAN") => {
                    self.expect_lparen()?;
                    while let Some(param_tok) = self.peek() {
                        if matches!(param_tok, Token::RParen) {
                            self.next();
                            break;
                        }
                        if matches!(param_tok, Token::Comma) {
                            self.next();
                            continue;
                        }
                        let key = match self.next() {
                            Some(Token::Word(k)) => k,
                            _ => return Err(AetherError::QueryParseError { code: 41 }),
                        };
                        match self.next() {
                            Some(Token::Equals) => {}
                            _ => return Err(AetherError::QueryParseError { code: 42 }),
                        }
                        let val = match self.next() {
                            Some(Token::Word(v)) => v,
                            _ => return Err(AetherError::QueryParseError { code: 43 }),
                        };
                        if key.eq_ignore_ascii_case("dim") {
                            euclidean_dim = val
                                .parse::<usize>()
                                .map_err(|_| AetherError::QueryParseError { code: 44 })?;
                        }
                    }
                }
                _ => return Err(AetherError::QueryParseError { code: 45 }),
            }
        }
        Err(AetherError::QueryParseError { code: 46 })
    }

    fn parse_insert_vertex(&mut self) -> Result<AqlStatement, AetherError> {
        self.expect_word("VERTEX")?;
        let vertex_id = match self.next() {
            Some(Token::Word(w)) => w
                .parse::<u64>()
                .map_err(|_| AetherError::QueryParseError { code: 50 })?,
            _ => return Err(AetherError::QueryParseError { code: 51 }),
        };

        self.expect_word("VECTOR")?;
        let vector = self.parse_f32_list()?;

        self.expect_word("STALK")?;
        let stalk_vec = self.parse_f32_list()?;
        if stalk_vec.len() != 8 {
            return Err(AetherError::QueryParseError { code: 52 });
        }
        let mut stalk = [0.0f32; 8];
        for (i, val) in stalk_vec.iter().enumerate().take(8) {
            if let Some(slot) = stalk.get_mut(i) {
                *slot = *val;
            }
        }

        self.expect_word("NEIGHBORS")?;
        let neighbors = self.parse_u64_list()?;

        Ok(AqlStatement::InsertVertex {
            vertex_id,
            vector,
            stalk,
            neighbors,
        })
    }

    fn parse_traverse(&mut self) -> Result<AqlStatement, AetherError> {
        self.expect_word("FROM")?;
        let start_vertex = match self.next() {
            Some(Token::Word(w)) => w
                .parse::<u64>()
                .map_err(|_| AetherError::QueryParseError { code: 60 })?,
            _ => return Err(AetherError::QueryParseError { code: 61 }),
        };

        self.expect_word("HOPS")?;
        let hops = match self.next() {
            Some(Token::Word(w)) => w
                .parse::<usize>()
                .map_err(|_| AetherError::QueryParseError { code: 62 })?,
            _ => return Err(AetherError::QueryParseError { code: 63 }),
        };

        self.expect_word("USING")?;
        self.expect_word("SEMIRING")?;

        let semiring = match self.next() {
            Some(Token::Word(w)) if w.eq_ignore_ascii_case("TROPICAL") => SemiringKind::Tropical,
            Some(Token::Word(w)) if w.eq_ignore_ascii_case("DIFFUSION") => SemiringKind::Diffusion,
            Some(Token::Word(w)) if w.eq_ignore_ascii_case("BOTTLENECK") => {
                SemiringKind::Bottleneck
            }
            _ => return Err(AetherError::QueryParseError { code: 64 }),
        };

        let mut beta = 1.0f32;
        let mut gamma = 1.0f32;
        let mut limit = 10usize;

        while let Some(tok) = self.next() {
            match tok {
                Token::Word(w) if w.eq_ignore_ascii_case("WHERE") => {
                    // WHERE clause: BETA = <val> [AND GAMMA = <val>]
                    while let Some(clause_tok) = self.peek() {
                        match clause_tok {
                            Token::Word(k) if k.eq_ignore_ascii_case("LIMIT") => break,
                            Token::Word(k) if k.eq_ignore_ascii_case("AND") => {
                                self.next();
                                continue;
                            }
                            Token::Word(k) if k.eq_ignore_ascii_case("BETA") => {
                                self.next();
                                match self.next() {
                                    Some(Token::Equals) => {}
                                    _ => return Err(AetherError::QueryParseError { code: 65 }),
                                }
                                let val_str = match self.next() {
                                    Some(Token::Word(v)) => v,
                                    _ => return Err(AetherError::QueryParseError { code: 66 }),
                                };
                                beta = val_str
                                    .parse::<f32>()
                                    .map_err(|_| AetherError::QueryParseError { code: 67 })?;
                            }
                            Token::Word(k) if k.eq_ignore_ascii_case("GAMMA") => {
                                self.next();
                                match self.next() {
                                    Some(Token::Equals) => {}
                                    _ => return Err(AetherError::QueryParseError { code: 68 }),
                                }
                                let val_str = match self.next() {
                                    Some(Token::Word(v)) => v,
                                    _ => return Err(AetherError::QueryParseError { code: 69 }),
                                };
                                gamma = val_str
                                    .parse::<f32>()
                                    .map_err(|_| AetherError::QueryParseError { code: 70 })?;
                            }
                            _ => break,
                        }
                    }
                }
                Token::Word(w) if w.eq_ignore_ascii_case("LIMIT") => {
                    let limit_str = match self.next() {
                        Some(Token::Word(v)) => v,
                        _ => return Err(AetherError::QueryParseError { code: 71 }),
                    };
                    limit = limit_str
                        .parse::<usize>()
                        .map_err(|_| AetherError::QueryParseError { code: 72 })?;
                }
                _ => return Err(AetherError::QueryParseError { code: 73 }),
            }
        }

        Ok(AqlStatement::Traverse {
            start_vertex,
            hops,
            semiring,
            beta,
            gamma,
            limit,
        })
    }

    fn parse_upsert_vertex(&mut self) -> Result<AqlStatement, AetherError> {
        self.expect_word("VERTEX")?;
        let vertex_id = match self.next() {
            Some(Token::Word(w)) => w
                .parse::<u64>()
                .map_err(|_| AetherError::QueryParseError { code: 50 })?,
            _ => return Err(AetherError::QueryParseError { code: 51 }),
        };

        let mut manifold = None;
        if let Some(Token::Word(w)) = self.peek() {
            if w.eq_ignore_ascii_case("INTO") {
                self.next();
                if let Some(Token::Word(m)) = self.next() {
                    manifold = Some(m.to_string());
                }
            }
        }

        let mut vector = Vec::new();
        let mut stalk = [0.0f32; 8];
        let mut neighbors = Vec::new();

        while let Some(tok) = self.next() {
            match tok {
                Token::Word(w) if w.eq_ignore_ascii_case("SET") => continue,
                Token::Word(w)
                    if w.eq_ignore_ascii_case("VECTOR")
                        || w.eq_ignore_ascii_case("COORDINATES") =>
                {
                    if let Some(Token::Equals) = self.peek() {
                        self.next();
                    }
                    vector = self.parse_f32_list()?;
                }
                Token::Word(w) if w.eq_ignore_ascii_case("STALK") => {
                    if let Some(Token::Equals) = self.peek() {
                        self.next();
                    }
                    let stalk_vec = self.parse_f32_list()?;
                    for (i, val) in stalk_vec.iter().enumerate().take(8) {
                        if let Some(slot) = stalk.get_mut(i) {
                            *slot = *val;
                        }
                    }
                }
                Token::Word(w) if w.eq_ignore_ascii_case("NEIGHBORS") => {
                    if let Some(Token::Equals) = self.peek() {
                        self.next();
                    }
                    neighbors = self.parse_u64_list()?;
                }
                _ => return Err(AetherError::QueryParseError { code: 53 }),
            }
        }

        Ok(AqlStatement::UpsertVertex {
            vertex_id,
            manifold,
            vector,
            stalk,
            neighbors,
        })
    }

    fn parse_connect_vertex(&mut self) -> Result<AqlStatement, AetherError> {
        self.expect_word("VERTEX")?;
        let source = match self.next() {
            Some(Token::Word(w)) => w
                .parse::<u64>()
                .map_err(|_| AetherError::QueryParseError { code: 80 })?,
            _ => return Err(AetherError::QueryParseError { code: 81 }),
        };

        self.expect_word("TO")?;
        let target = match self.next() {
            Some(Token::Word(w)) => w
                .parse::<u64>()
                .map_err(|_| AetherError::QueryParseError { code: 82 })?,
            _ => return Err(AetherError::QueryParseError { code: 83 }),
        };

        let mut manifold = None;
        let mut morphism_type = "IDENTITY".to_string();
        let mut weight = 1.0f32;

        while let Some(tok) = self.next() {
            match tok {
                Token::Word(w) if w.eq_ignore_ascii_case("INTO") => {
                    if let Some(Token::Word(m)) = self.next() {
                        manifold = Some(m.to_string());
                    }
                }
                Token::Word(w)
                    if w.eq_ignore_ascii_case("WITH") || w.eq_ignore_ascii_case("USING") =>
                {
                    continue;
                }
                Token::Word(w) if w.eq_ignore_ascii_case("MORPHISM") => {
                    if let Some(Token::Word(m)) = self.next() {
                        morphism_type = m.to_string();
                    }
                }
                Token::Word(w) if w.eq_ignore_ascii_case("WEIGHT") => {
                    if let Some(Token::Equals) = self.peek() {
                        self.next();
                    }
                    if let Some(Token::Word(v)) = self.next() {
                        weight = v
                            .parse::<f32>()
                            .map_err(|_| AetherError::QueryParseError { code: 84 })?;
                    }
                }
                _ => return Err(AetherError::QueryParseError { code: 85 }),
            }
        }

        Ok(AqlStatement::ConnectVertex {
            source,
            target,
            manifold,
            morphism_type,
            weight,
        })
    }

    fn parse_optimize_manifold(&mut self) -> Result<AqlStatement, AetherError> {
        self.expect_word("MANIFOLD")?;
        let manifold = match self.next() {
            Some(Token::Word(w)) => w.to_string(),
            _ => return Err(AetherError::QueryParseError { code: 90 }),
        };

        let mut from_level = 1u8;
        let mut to_level = 2u8;
        let mut method = "VORONOI_BOUNDARY_CERTIFICATES".to_string();
        let mut return_telemetry = false;

        while let Some(tok) = self.next() {
            match tok {
                Token::Word(w)
                    if w.eq_ignore_ascii_case("COMPACT") || w.eq_ignore_ascii_case("FROM") =>
                {
                    if let Some(Token::Word(sub)) = self.peek() {
                        if sub.eq_ignore_ascii_case("LEVEL") {
                            self.next();
                        }
                    }
                    let from_str = match self.next() {
                        Some(Token::Word(s)) => s,
                        _ => return Err(AetherError::QueryParseError { code: 91 }),
                    };
                    from_level = if from_str.starts_with('L') || from_str.starts_with('l') {
                        match from_str.get(1..) {
                            Some(tail) => tail.parse::<u8>().unwrap_or(1),
                            None => 1,
                        }
                    } else {
                        from_str.parse::<u8>().unwrap_or(1)
                    };

                    self.expect_word("TO")?;
                    if let Some(Token::Word(sub)) = self.peek() {
                        if sub.eq_ignore_ascii_case("LEVEL") {
                            self.next();
                        }
                    }
                    let to_str = match self.next() {
                        Some(Token::Word(s)) => s,
                        _ => return Err(AetherError::QueryParseError { code: 92 }),
                    };
                    to_level = if to_str.starts_with('L') || to_str.starts_with('l') {
                        match to_str.get(1..) {
                            Some(tail) => tail.parse::<u8>().unwrap_or(2),
                            None => 2,
                        }
                    } else {
                        to_str.parse::<u8>().unwrap_or(2)
                    };
                }
                Token::Word(w)
                    if w.eq_ignore_ascii_case("USING") || w.eq_ignore_ascii_case("WITH") =>
                {
                    if let Some(Token::Word(sub)) = self.peek() {
                        if sub.eq_ignore_ascii_case("METHOD")
                            || sub.eq_ignore_ascii_case("STRATEGY")
                        {
                            self.next();
                        }
                    }
                    if let Some(Token::Word(m)) = self.next() {
                        method = m.to_string();
                    }
                }
                Token::Word(w) if w.eq_ignore_ascii_case("RETURN") => {
                    if let Some(Token::Word(t)) = self.next() {
                        if t.eq_ignore_ascii_case("TELEMETRY") {
                            return_telemetry = true;
                        }
                    }
                }
                _ => return Err(AetherError::QueryParseError { code: 93 }),
            }
        }

        Ok(AqlStatement::OptimizeManifold {
            manifold,
            from_level,
            to_level,
            method,
            return_telemetry,
        })
    }
}

/// Parses a declarative AQL string into a strongly-typed `AqlStatement` AST.
///
/// Supports case-insensitive keywords and flexible whitespace.
/// Returns `Err(AetherError::QueryParseError)` on syntax faults.
pub fn parse_aql(input: &str) -> Result<AqlStatement, AetherError> {
    let tokens = tokenize(input);
    if tokens.is_empty() {
        return Err(AetherError::QueryParseError { code: 1 });
    }

    let mut parser = Parser::new(tokens);
    match parser.next() {
        Some(Token::Word(w)) if w.eq_ignore_ascii_case("CREATE") => parser.parse_create_manifold(),
        Some(Token::Word(w)) if w.eq_ignore_ascii_case("INSERT") => parser.parse_insert_vertex(),
        Some(Token::Word(w)) if w.eq_ignore_ascii_case("UPSERT") => parser.parse_upsert_vertex(),
        Some(Token::Word(w)) if w.eq_ignore_ascii_case("CONNECT") => parser.parse_connect_vertex(),
        Some(Token::Word(w)) if w.eq_ignore_ascii_case("OPTIMIZE") => {
            parser.parse_optimize_manifold()
        }
        Some(Token::Word(w)) if w.eq_ignore_ascii_case("TRAVERSE") => parser.parse_traverse(),
        _ => Err(AetherError::QueryParseError { code: 2 }),
    }
}

/// Compilation helper mapping declarative AQL statements directly into low-level query plans.
pub struct AqlCompileHelper;

impl AqlCompileHelper {
    /// Compiles an `AqlStatement::Traverse` into a concrete `QueryPlan`.
    pub fn compile_traverse_plan(stmt: &AqlStatement) -> Result<(u64, QueryPlan), AetherError> {
        match stmt {
            AqlStatement::Traverse {
                start_vertex,
                hops,
                semiring,
                beta,
                gamma,
                limit,
            } => {
                let plan = QueryPlan {
                    semiring_mode: (*semiring).into(),
                    max_hops: *hops,
                    top_k: *limit,
                    beam_width: (*limit).max(64),
                    beta: *beta,
                    gamma: *gamma,
                    prune_threshold: 1e-4,
                };
                Ok((*start_vertex, plan))
            }
            _ => Err(AetherError::QueryParseError { code: 80 }),
        }
    }

    /// Executes an `AqlStatement::Traverse` directly onto `SemiringSpMvEngine::execute_query`
    /// with strictly zero hot-path heap allocations inside `HotPathAllocGuard` (AGENTS.md Rule 3.1).
    pub fn execute_traverse(
        stmt: &AqlStatement,
        page_lookup: impl Fn(u64) -> Option<&'static TensorGraphPage>,
        distance_oracle: impl Fn(u64) -> f32,
        results: &mut [FrontierNode; MAX_QUERY_FRONTIER],
        result_count: &mut usize,
    ) -> Result<(), AetherError> {
        let (start_vertex, plan) = Self::compile_traverse_plan(stmt)?;
        SemiringSpMvEngine::execute_query(
            &plan,
            &[start_vertex],
            page_lookup,
            distance_oracle,
            results,
            result_count,
        )
    }
}
