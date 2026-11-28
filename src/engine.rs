use std::io;
use std::io::Write;
use std::time::{Duration, Instant};
use shakmaty::{Chess, Position, Move, Color, MoveList, Role, PlayError, EnPassantMode, zobrist::Zobrist64, CastlingMode, PositionError};
use rand;
use crate::constants::{INF, TABLE_SIZE, PHASE_MAX, NMP, LMR, LMR_T1, piece_value, piece_square_value, king_pawn_distance, phase};
use crate::opening_book::OpeningBook;
use crate::transition_table::{TranspositionTable, EntryType, TTEntry};

// mate score: a mate delivered at ply p has score MATE - p (so faster mates score higher)
const MATE: i32 = 1_000_000;
const MATE_BOUND: i32 = MATE - 1_000;
const MAX_PLY: usize = 128;
// quiescence: captures that cannot reach alpha even with this extra margin are skipped
const DELTA_MARGIN: i32 = 200;
// history values must stay below the killer score used in move ordering
const HISTORY_MAX: i32 = 5_000_000;

const SCORE_TT_MOVE: i32 = 1_000_000_000;
const SCORE_CAPTURE: i32 = 10_000_000;
const SCORE_PROMOTION: i32 = 9_500_000;
const SCORE_KILLER_1: i32 = 9_000_000;
const SCORE_KILLER_2: i32 = 8_900_000;

fn score_to_tt(score: i32, ply: i32) -> i32 {
    if score >= MATE_BOUND { score + ply } else if score <= -MATE_BOUND { score - ply } else { score }
}

fn score_from_tt(score: i32, ply: i32) -> i32 {
    if score >= MATE_BOUND { score - ply } else if score <= -MATE_BOUND { score + ply } else { score }
}

#[derive(Clone)]
pub struct Game {
    history: Vec<Chess>,
    hashes: Vec<Zobrist64>,
    max_moves: Option<i32>,
}

impl Game {
    pub fn new() -> Self {
        Game {
            history: vec![Chess::default()],
            hashes: vec![Chess::default().zobrist_hash(EnPassantMode::Legal)],
            max_moves: None,
        }
    }

    pub fn from_pos(c: Chess) -> Self {
        let hash: Zobrist64 = c.zobrist_hash(EnPassantMode::Legal);
        Game {
            history: vec![c],
            hashes: vec![hash],
            max_moves: None,
        }
    }

    pub fn set_max_moves(&mut self, max_moves: Option<i32>) {
        self.max_moves = max_moves;
    }

    pub fn get_max_moves(&self) -> Option<i32> {
        self.max_moves
    }

    pub fn current(&self) -> &Chess {
        self.history.last().unwrap()
    }

    // hash of the current position (already computed by push / null_move)
    pub fn hash(&self) -> Zobrist64 {
        *self.hashes.last().unwrap()
    }

    pub fn len(&self) -> usize {
        self.history.len()
    }

    // checked version, used for moves coming from outside (GUI)
    pub fn push(&mut self, mv: Move) -> Result<(), PlayError<Chess>> {
        let next = self.current().clone().play(mv)?;
        let hash = next.zobrist_hash::<Zobrist64>(EnPassantMode::Legal);
        self.history.push(next);
        self.hashes.push(hash);
        Ok(())
    }

    // unchecked version, used in the search (moves come from legal_moves())
    pub fn push_unchecked(&mut self, mv: Move) {
        let mut next = self.current().clone();
        next.play_unchecked(mv);
        let hash = next.zobrist_hash::<Zobrist64>(EnPassantMode::Legal);
        self.history.push(next);
        self.hashes.push(hash);
    }

    pub fn pop(&mut self) -> Option<Chess> {
        if self.history.len() > 1 {
            self.hashes.pop();
            self.history.pop()
        } else {
            None
        }
    }

    pub fn count_repetitions(&self, r: usize) -> bool {
        let n = self.hashes.len();
        let back = (self.current().halfmoves() as usize).min(n - 1);
        let cur = self.hashes[n - 1];
        let count = 1 + (2..=back).step_by(2).filter(|&i| self.hashes[n - 1 - i] == cur).count();
        count >= r
    }

    pub fn maxed_moves(&self) -> bool {
        match self.max_moves {
            Some(max_moves) => self.current().halfmoves() >= max_moves as u32,
            None => false,
        }
    }

    pub fn get_phase(&self) -> f32 {
        phase(self.current().board())
    }

    pub fn is_endgame(&self) -> bool {
        if self.get_phase() <= 0.33 { true } else { false }
    }

    pub fn null_move(&mut self) -> Result<(), PositionError<Chess>> {
        let null_pos = self.current().clone().swap_turn()?;
        let hash = null_pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal);
        self.history.push(null_pos);
        self.hashes.push(hash);
        Ok(())
    }
}

#[derive(Clone)]
pub struct SearchResult {
    pub best_move: Move,
    pub score: i32,
}

#[derive(Clone, Copy, Default)]
pub struct SearchStats {
    pub nodes: u64,
    pub q_nodes: u64,
    pub tt_hits: u64,
    pub nmp_tried: u64,
    pub nmp_succ: u64,
    pub lmr_tried: u64,
    pub lmr_succ: u64,
    
    // beta cutoffs, and how many of them happened on the first move (ordering quality)
    pub cutoffs: u64,
    pub first_cutoffs: u64,
}

pub struct Engine {
    pub game: Game,
    color: Color,
    max_depth: i32,

    soft_timer: Duration,
    hard_timer: Duration,
    stopped: bool,
    time_check: u32,

    best_move: Option<Move>,
    best_score: i32,
    start_timer: Instant,
    hash_table: TranspositionTable,
    q_hash_table: TranspositionTable,
    silent_history: [[i32; 64]; 64],
    killers: [[Option<Move>; 2]; MAX_PLY],
    lmr_table: [[i32; 64]; 64],

    stats: SearchStats,
}

impl Engine {
    pub fn new(g: Game, c: Color) -> Self {
        let mut lmr_table = [[0i32; 64]; 64];
        for d in 1..64 {
            for i in 1..64 {
                lmr_table[d][i] = (0.75 + (d as f64).ln() * (i as f64).ln() / 2.25) as i32;
            }
        }

        Engine {
            game: g,
            color: c,
            max_depth: 30,

            soft_timer: Duration::from_secs(30),
            hard_timer: Duration::from_secs(30),
            stopped: false,
            time_check: 0,

            best_move: None,
            best_score: 0,
            start_timer: Instant::now(),
            hash_table: TranspositionTable::new(TABLE_SIZE),
            q_hash_table: TranspositionTable::new(TABLE_SIZE),
            silent_history: [[0; 64]; 64],
            killers: [[None; 2]; MAX_PLY],
            lmr_table,

            stats: SearchStats::default(),
        }
    }

    pub fn set_game(&mut self, game: Game) {
        self.game = game;
        self.best_move = None;
        self.best_score = 0;
    }

    pub fn set_max_depth(&mut self, max_depth: i32) {
        self.max_depth = max_depth;
    }

    pub fn set_timer(&mut self, soft: Duration, hard: Duration) {
        self.soft_timer = soft;
        self.hard_timer = hard.max(soft);
    }

    pub fn set_color(&mut self, color: Color) {
        self.color = color;
    }

    pub fn get_color(&self) -> Color {
        self.color
    }

    pub fn get_best_score(&self) -> i32 {
        self.best_score
    }

    pub fn get_stats(&self) -> SearchStats {
        self.stats
    }

    pub fn get_search_stats(&self) -> &SearchStats {
        &self.stats
    }

    pub fn is_timeout(&mut self) -> bool {
        if self.stopped {
            return true;
        }
        self.time_check += 1;
        if self.time_check >= 2048 {
            self.time_check = 0;
            if self.start_timer.elapsed() >= self.hard_timer {
                self.stopped = true;
            }
        }
        self.stopped
    }

    fn print_info(&self, mut depth: i32) {
        let mut pv_str = shakmaty::uci::UciMove::from_move(self.best_move.unwrap(), CastlingMode::Standard).to_string();
        if depth > 0 {
            let pv = self.extract_pv(depth);
            if !pv.is_empty() {
                pv_str = pv.iter()
                    .map(|m| shakmaty::uci::UciMove::from_move(*m, CastlingMode::Standard).to_string())
                    .collect::<Vec<_>>()
                    .join(" ");
            }
        } else {
            // move from opening book
            depth = 1;
        }

        let score_str = if self.best_score.abs() >= MATE_BOUND {
            let moves = (MATE - self.best_score.abs() + 1) / 2;
            format!("mate {}", if self.best_score > 0 { moves } else { -moves })
        } else {
            format!("cp {}", self.best_score)
        };

        let nodes = self.stats.nodes + self.stats.q_nodes + self.stats.tt_hits;
        let ms = self.start_timer.elapsed().as_millis().max(1);

        println!("info depth {} score {} nodes {} nps {} time {} pv {}",
                 depth,
                 score_str,
                 nodes,
                 (nodes as u128) * 1000 / ms,
                 ms,
                 pv_str
        );
        io::stdout().flush().unwrap();
    }

    fn extract_pv(&self, depth: i32) -> Vec<Move> {
        let mut pv = Vec::new();
        let mut game = self.game.clone();

        for _ in 0..depth {
            let h = game.hash();
            if let Some(entry) = self.hash_table.get(h) {
                if let Some(mv) = entry.best_move {
                    if game.current().legal_moves().contains(&mv) {
                        pv.push(mv);
                        game.push_unchecked(mv);
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        pv
    }

    pub fn play(&mut self, opening_book: &Option<OpeningBook>, print_info: bool) -> SearchResult {

        // search opening book
        if let Some(book) = opening_book {
            let hash = u64::from(self.game.hash());

            // keep only the book moves that can be decoded in this position
            let candidates: Vec<(Move, u32)> = book.get_moves(hash)
                .iter()
                .filter_map(|e| e.decode_move(self.game.current()).map(|m| (m, e.weight as u32)))
                .collect();
            let tot_weight: u32 = candidates.iter().map(|c| c.1).sum();

            if tot_weight > 0 {
                // select random move from book
                // probability based on "weight" of move in the book
                let mut choice = rand::random_range(0..tot_weight);

                for (mv, weight) in candidates {
                    if choice < weight {
                        self.best_move = Some(mv);
                        self.game.push_unchecked(mv);
                        // after the move the side to move is the opponent
                        self.best_score = -self.eval();
                        self.game.pop();
                        if print_info {
                            self.print_info(0);
                        }
                        return SearchResult { best_move: mv, score: self.best_score };
                    }
                    choice -= weight;
                }
            }
        }

        self.best_move = None;
        self.best_score = 0;
        self.stats = SearchStats::default();
        self.killers = [[None; 2]; MAX_PLY];
        // age the history between searches
        for row in self.silent_history.iter_mut() {
            for v in row.iter_mut() {
                *v /= 2;
            }
        }

        let mut prev_res: Vec<SearchResult> = vec![];

        // start move timer
        self.start_timer = Instant::now();
        self.stopped = false;
        self.time_check = 0;

        let only_move = self.game.current().legal_moves().len() <= 1;
        let mut stable = 0;
        let mut prev_best: Option<Move> = None;
        let mut prev_score = 0;
        let mut prev_elapsed = 0u64;

        // with iterative deepening
        for depth in 1..=self.max_depth {

            let score = if depth <= 2 {
                // first two iterations have a "full" window
                self.negamax(depth, 0, -INF, INF, false)
            } else {
                // use aspiration window on later iterations
                let mut delta = 50;
                let last = prev_res[depth as usize - 2].score;
                let mut alpha = (last - delta).max(-INF);
                let mut beta = (last + delta).min(INF);

                loop {
                    match self.negamax(depth, 0, alpha, beta, false) {
                        None => break None,
                        Some(s) if s <= alpha => {
                            alpha = (alpha - delta).max(-INF);
                            delta *= 2;
                        }
                        Some(s) if s >= beta => {
                            beta = (beta + delta).min(INF);
                            delta *= 2;
                        }
                        Some(s) => break Some(s),
                    }
                }
            };

            // iteration interrupted, use last valid
            let Some(score) = score else {
                return prev_res.last().cloned().unwrap_or_else(|| SearchResult {
                    best_move: self.game.current().legal_moves()[0],
                    score: 0,
                });
            };

            self.best_score = score;

            prev_res.push(SearchResult {
                best_move: self.best_move.unwrap(),
                score: self.best_score,
            });

            if print_info {
                self.print_info(depth);
            }

            // only one move available or no moves at all, or a forced mate was found
            if only_move || self.best_score.abs() >= MATE_BOUND {
                break;
            }

            // dynamic soft timer
            if depth > 1 && self.best_move == prev_best { stable += 1 } else { stable = 0 }

            let mut target = self.soft_timer.as_millis() as u64;
            target = match stable {
                0 => target * 13 / 10,
                1 | 2 => target,
                _ => target * 6 / 10,
            };
            if depth > 1 && prev_score - self.best_score > 30 {
                target = target * 15 / 10;
            }
            target = target.min(self.hard_timer.as_millis() as u64);

            // do not start an iteration that is not expected to finish in time
            // (an iteration costs about 3 times the previous one)
            let elapsed = self.start_timer.elapsed().as_millis() as u64;
            let last_iter = elapsed - prev_elapsed;
            prev_elapsed = elapsed;
            if elapsed + last_iter * 3 > target {
                break;
            }

            prev_best = self.best_move;
            prev_score = self.best_score;
        }
        SearchResult {
            best_move: self.best_move.unwrap(),
            score: self.best_score,
        }
    }

    fn score_moves(&self, moves: &MoveList, tt_move: Option<Move>, ply: usize, scores: &mut [i32; 256]) {
        for (i, m) in moves.iter().enumerate() {
            scores[i] = if Some(*m) == tt_move {
                SCORE_TT_MOVE
            } else if m.is_capture() {
                // MVV-LVA
                SCORE_CAPTURE + piece_value(m.capture().unwrap()) * 10 - piece_value(m.role())
            } else if let Some(promotion) = m.promotion() {
                SCORE_PROMOTION + piece_value(promotion)
            } else if self.killers[ply][0] == Some(*m) {
                SCORE_KILLER_1
            } else if self.killers[ply][1] == Some(*m) {
                SCORE_KILLER_2
            } else {
                self.silent_history[m.from().unwrap() as usize][m.to() as usize]
            };
        }
    }

    // selection of the best remaining move: most of the time a cutoff happens after few moves, so a full sort would be useless
    fn pick_next(moves: &mut MoveList, scores: &mut [i32; 256], start: usize) {
        let mut best = start;
        for j in (start + 1)..moves.len() {
            if scores[j] > scores[best] {
                best = j;
            }
        }
        if best != start {
            moves.swap(start, best);
            scores.swap(start, best);
        }
    }

    fn store_killer(&mut self, ply: i32, m: Move) {
        let p = ply as usize;
        if self.killers[p][0] != Some(m) {
            self.killers[p][1] = self.killers[p][0];
            self.killers[p][0] = Some(m);
        }
    }

    fn update_history(&mut self, m: Move, depth: i32) {
        let v = &mut self.silent_history[m.from().unwrap() as usize][m.to() as usize];
        *v = (*v + depth * depth).min(HISTORY_MAX);
    }

    // white minus black
    fn eval_white(&self) -> i32 {
        let board = self.game.current().board();
        let phase = self.game.get_phase();
        let mut score = 0;

        for color in [Color::White, Color::Black] {
            let mut value = 0;
            for role in [Role::Pawn, Role::Knight, Role::Bishop, Role::Rook, Role::Queen, Role::King] {
                let mut pieces = board.by_color(color).intersect(board.by_role(role));
                while let Some(square) = pieces.pop_front() {
                    value += piece_square_value(role, color, square, board, phase);
                }
            }
            value += king_pawn_distance(color, board, phase);
            if color == Color::White {
                score += value;
            } else {
                score -= value;
            }
        }
        score
    }

    // score from the point of view of the side to move
    fn eval(&self) -> i32 {
        let e = self.eval_white();
        if self.game.current().turn() == Color::White { e } else { -e }
    }

    // zugzwang guard for null move pruning: the side to move needs pieces other than pawns and king
    fn has_non_pawn_material(&self) -> bool {
        let pos = self.game.current();
        let board = pos.board();
        let mine = board.by_color(pos.turn());
        mine.count() > (mine & (board.pawns() | board.kings())).count()
    }

    fn is_draw(&self, null_move: bool) -> bool {
        // a position that already appeared once is scored as a draw inside the search
        // (the position after a null move can coincide with a real one, so skip it)
        if !null_move && self.game.count_repetitions(2) {
            return true;
        }
        self.game.maxed_moves() && !self.game.current().is_checkmate()
    }

    // ---- search ----

    fn quiescence(&mut self, mut alpha: i32, beta: i32, ply: i32) -> Option<i32> {

        // check timer
        if self.is_timeout() {
            return None;
        }
        self.stats.q_nodes += 1;

        if ply >= MAX_PLY as i32 - 1 {
            return Some(self.eval());
        }

        // memoization with Zobrist hashes (qsearch entries are all stored with depth 0)
        let h = self.game.hash();
        let mut tt_move: Option<Move> = None;

        if let Some(entry) = self.q_hash_table.get(h) {
            tt_move = entry.best_move;
            let s = score_from_tt(entry.score, ply);
            let cut = match entry.entry_type {
                EntryType::Exact => true,
                EntryType::LowerBound => s >= beta,
                EntryType::UpperBound => s <= alpha,
            };
            if cut {
                self.stats.tt_hits += 1;
                return Some(s);
            }
        }

        let in_check = self.game.current().is_check();
        let alpha_orig = alpha;

        let mut best_score;
        let mut stand_pat = -INF;

        if in_check {
            // no stand pat when in check: every evasion must be searched
            best_score = -MATE + ply;
        } else {
            stand_pat = self.eval();
            if stand_pat >= beta {
                return Some(stand_pat);
            }
            best_score = stand_pat;
            if stand_pat > alpha {
                alpha = stand_pat;
            }
        }

        let mut moves = self.game.current().legal_moves();
        if !in_check {
            moves.retain(|m| m.is_capture() || m.promotion() == Some(Role::Queen));
        }

        if moves.is_empty() {
            return if in_check { Some(-MATE + ply) } else { Some(stand_pat) };
        }

        let mut scores = [0i32; 256];
        self.score_moves(&moves, tt_move, ply as usize, &mut scores);

        let mut best_move: Option<Move> = None;

        for i in 0..moves.len() {
            Self::pick_next(&mut moves, &mut scores, i);
            let m = moves[i];

            // delta pruning
            if !in_check && !m.is_promotion() {
                if let Some(victim) = m.capture() {
                    if stand_pat + piece_value(victim) + DELTA_MARGIN <= alpha {
                        continue;
                    }
                }
            }

            self.game.push_unchecked(m);
            let res = self.quiescence(-beta, -alpha, ply + 1);
            self.game.pop();
            let score = -res?;

            if score > best_score {
                best_score = score;
                best_move = Some(m);
                if score > alpha {
                    alpha = score;
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }

        let entry_type = if best_score >= beta {
            EntryType::LowerBound
        } else if best_score > alpha_orig {
            EntryType::Exact
        } else {
            EntryType::UpperBound
        };

        self.q_hash_table.insert(TTEntry {
            hash: h,
            depth: 0,
            score: score_to_tt(best_score, ply),
            entry_type,
            best_move: if entry_type == EntryType::UpperBound { tt_move } else { best_move },
        });

        Some(best_score)
    }

    // search of one child (the move is already on the board):
    // first move with a full window, the others with a null window (PVS),
    // possibly reduced (LMR) and re-searched if they turn out to be interesting
    fn search_child(&mut self, depth: i32, ply: i32, alpha: i32, beta: i32, first: bool, reduction: i32) -> Option<i32> {
        if first {
            return Some(-self.negamax(depth - 1, ply + 1, -beta, -alpha, false)?);
        }

        let mut score = -self.negamax(depth - 1 - reduction, ply + 1, -alpha - 1, -alpha, false)?;

        if reduction > 0 {
            self.stats.lmr_tried += 1;
            if score <= alpha {
                self.stats.lmr_succ += 1;
            }
        }

        // the reduced search says the move is good: verify it at full depth
        if score > alpha && reduction > 0 {
            score = -self.negamax(depth - 1, ply + 1, -alpha - 1, -alpha, false)?;
        }

        // inside the window: full window search
        if score > alpha && score < beta {
            score = -self.negamax(depth - 1, ply + 1, -beta, -alpha, false)?;
        }

        Some(score)
    }

    fn negamax(&mut self, mut depth: i32, ply: i32, mut alpha: i32, beta: i32, null_move: bool) -> Option<i32> {

        // check timer
        if self.is_timeout() {
            return None;
        }

        let is_root = ply == 0;
        let is_pv = beta - alpha > 1;

        // draws (never at the root: the search must always return a move)
        if !is_root && self.is_draw(null_move) {
            return Some(0);
        }

        if ply >= MAX_PLY as i32 - 1 {
            return Some(self.eval());
        }

        // check extension
        let in_check = self.game.current().is_check();
        if in_check {
            depth += 1;
        }

        if depth <= 0 {
            return self.quiescence(alpha, beta, ply);
        }

        self.stats.nodes += 1;

        // memoization with Zobrist hashes
        let h = self.game.hash();
        let mut tt_move: Option<Move> = None;

        if let Some(entry) = self.hash_table.get(h) {
            tt_move = entry.best_move;
            if !is_root && entry.depth >= depth {
                let s = score_from_tt(entry.score, ply);
                let cut = match entry.entry_type {
                    EntryType::Exact => true,
                    EntryType::LowerBound => s >= beta,
                    EntryType::UpperBound => s <= alpha,
                };
                if cut {
                    self.stats.tt_hits += 1;
                    return Some(s);
                }
            }
        }

        // null move pruning (NMP)
        if !null_move && !is_pv && !in_check && depth >= NMP
            && beta.abs() < MATE_BOUND
            && self.has_non_pawn_material()
            && self.eval() >= beta
        {
            if self.game.null_move().is_ok() {
                let r = 3 + depth / 6;
                let res = self.negamax(depth - 1 - r, ply + 1, -beta, -beta + 1, true);
                self.game.pop();
                let score = -res?;
                self.stats.nmp_tried += 1;
                if score >= beta {
                    self.stats.nmp_succ += 1;
                    return Some(beta);
                }
            }
        }

        let mut moves = self.game.current().legal_moves();

        // checkmate or stalemate
        if moves.is_empty() {
            return Some(if in_check { -MATE + ply } else { 0 });
        }

        let mut scores = [0i32; 256];
        self.score_moves(&moves, tt_move, ply as usize, &mut scores);

        let alpha_orig = alpha;
        let mut best_score = -INF;
        let mut best_move: Option<Move> = None;

        for i in 0..moves.len() {
            Self::pick_next(&mut moves, &mut scores, i);
            let m = moves[i];
            let is_quiet = !m.is_capture() && !m.is_promotion();

            self.game.push_unchecked(m);
            let gives_check = self.game.current().is_check();

            // late move reduction (LMR)
            let mut reduction = 0;
            if i as i32 >= LMR_T1 && depth >= LMR && is_quiet && !in_check && !gives_check {
                let r = self.lmr_table[(depth as usize).min(63)][i.min(63)];
                reduction = (r - if is_pv { 1 } else { 0 }).clamp(0, depth - 2);
            }

            let res = self.search_child(depth, ply, alpha, beta, i == 0, reduction);
            self.game.pop();
            let score = res?;

            if score > best_score {
                best_score = score;
                best_move = Some(m);

                // at the root save the best move
                if is_root {
                    self.best_move = Some(m);
                }

                if score > alpha {
                    alpha = score;
                    if alpha >= beta {
                        self.stats.cutoffs += 1;
                        if i == 0 {
                            self.stats.first_cutoffs += 1;
                        }
                        if is_quiet {
                            self.store_killer(ply, m);
                            self.update_history(m, depth);
                        }
                        break;
                    }
                }
            }
        }

        let entry_type = if best_score >= beta {
            EntryType::LowerBound
        } else if best_score > alpha_orig {
            EntryType::Exact
        } else {
            EntryType::UpperBound
        };

        self.hash_table.insert(TTEntry {
            hash: h,
            depth,
            score: score_to_tt(best_score, ply),
            entry_type,
            best_move: if entry_type == EntryType::UpperBound { tt_move } else { best_move },
        });

        Some(best_score)
    }
}
