use shakmaty::Color;
use vampirc_uci::{parse_one, UciMessage, UciTimeControl};
use std::io::{self, BufRead, Write};
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
                let time_limit = match time_control {
                    Some(UciTimeControl::MoveTime(duration)) => duration,
                    Some(UciTimeControl::TimeLeft { white_time, black_time, .. }) => {
                        let t = if fishbait.get_color() == Color::White { white_time } else { black_time };
                        match t {
                            Some(t) => t / 15,
                            None => chrono::Duration::seconds(30),
                        }
                    }
                    _ => chrono::Duration::seconds(30),
                };
                let result = fishbait.search(time_limit.to_std().unwrap(), true);
                let uci_move = shakmaty::uci::UciMove::from_move(
                    result.best_move,
                    shakmaty::CastlingMode::Standard,
                );
                println!("bestmove {}", uci_move);
                io::stdout().flush().unwrap();
            }
            UciMessage::Quit => break,
            _ => {}
        }
    }
}