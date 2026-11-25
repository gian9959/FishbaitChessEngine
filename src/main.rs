use shakmaty::Color;
use vampirc_uci::{parse_one, UciMessage, UciTimeControl};
use std::io::{self, BufRead, Write};
use std::time::Duration;
use fishbait_engine::{Fishbait, OpeningBook};

fn main() {
    let stdin = io::stdin();
    let mut book = OpeningBook::from_embedded();
    let mut fishbait = Fishbait::new(Color::White, Some(100), Some(book));

    for line in stdin.lock().lines() {
        let msg = parse_one(&line.unwrap());
        match msg {
            UciMessage::Uci => {
                println!("id name Fishbait");
                println!("id author Gianmaria Vasino");
                println!("uciok");
                io::stdout().flush().unwrap();
            }
            UciMessage::IsReady => {
                println!("readyok");
                io::stdout().flush().unwrap();
            }
            UciMessage::UciNewGame => {
                book = OpeningBook::from_embedded();
                fishbait = Fishbait::new(Color::White, Some(100), Some(book));
            }
            UciMessage::Position { startpos, fen, moves } => {
                let fen_str = if startpos { None } else { fen.map(|f| f.to_string()) };
                let moves: Vec<String> = moves.iter().map(|m| m.to_string()).collect();
                if let Err(e) = fishbait.set_position(fen_str.as_deref(), &moves) {
                    eprintln!("position error: {}", e);
                }
            }
            UciMessage::Go { time_control, .. } => {
                fishbait.set_color_uci();
                let white = fishbait.get_color() == Color::White;

                let (soft, hard) = match time_control {
                    Some(UciTimeControl::MoveTime(d)) => {
                        let d = d.to_std().unwrap_or(Duration::from_millis(50));
                        (d, d)
                    }
                    Some(UciTimeControl::TimeLeft { white_time, black_time, white_increment, black_increment, moves_to_go }) => {
                        if white {
                            time_budget(white_time, white_increment, moves_to_go)
                        } else {
                            time_budget(black_time, black_increment, moves_to_go)
                        }
                    }
                    _ => (Duration::from_secs(30), Duration::from_secs(30)),
                };

                let result = fishbait.search(soft, hard, true);
                let uci_move = shakmaty::uci::UciMove::from_move(result.best_move, shakmaty::CastlingMode::Standard);
                println!("bestmove {}", uci_move);
                io::stdout().flush().unwrap();
            }
            UciMessage::Quit => break,
            _ => {}
        }
    }
}

fn time_budget(my_time: Option<chrono::Duration>, inc: Option<chrono::Duration>, mtg: Option<u8>) -> (Duration, Duration) {
    let my_ms = my_time.map_or(30_000, |d| d.num_milliseconds());
    let inc_ms = inc.map_or(0, |d| d.num_milliseconds());
    let overhead = 50;
    let avail = (my_ms - overhead).max(10);
    let mtg = mtg.unwrap_or(30).max(1) as i64;
    let soft = (avail / mtg + inc_ms * 3 / 4).min(avail / 3).max(10);
    let hard = (soft * 4).min(avail / 3).max(soft);
    (Duration::from_millis(soft as u64), Duration::from_millis(hard as u64))
}