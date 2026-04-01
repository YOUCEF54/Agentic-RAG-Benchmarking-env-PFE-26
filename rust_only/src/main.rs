mod embedder;
mod store;
mod llm;

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use pdf_oxide::PdfDocument;
use reqwest::Client;
use text_splitter::TextSplitter;

const NUM_RUNS: u32 = 100;
const CSV_FILE: &str = "profile_results.csv";
const CSV_HEADER: &str = "run,pdf_read_ms,chunking_ms,model_embedding_ms,db_insert_ms,search_ms\n";

const QUESTION: &str = "Question: According to the abstract, what specific type of \
    'framework' does this paper propose to support knowledge management and decision-making?";

#[tokio::main]
async fn main() {
    // ── Load .env (mirrors Python's load_dotenv()) ─────────────────────────────
    dotenvy::dotenv().ok();

    let api_key = std::env::var("API_KEY").expect("API_KEY not set in .env");
    let model_name = std::env::var("MODEL").expect("MODEL not set in .env");
    let pdf_folder = std::env::var("PDF_FOLDER").unwrap_or_else(|_| "./pdfs".to_string());

    // ── Collect all PDF paths from the folder (mirrors Python) ─────────────────
    let pdf_files: Vec<std::path::PathBuf> = std::fs::read_dir(&pdf_folder)
        .unwrap_or_else(|_| panic!("Cannot read PDF folder: {}", pdf_folder))
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension()?.to_ascii_lowercase() == "pdf" {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    if pdf_files.is_empty() {
        panic!("No PDF files found in folder: {}", pdf_folder);
    }

    println!(
        "Found {} PDF(s): {:?}",
        pdf_files.len(),
        pdf_files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy())
            .collect::<Vec<_>>()
    );

    // ── CSV setup ──────────────────────────────────────────────────────────────
    if !Path::new(CSV_FILE).exists() {
        let mut f = OpenOptions::new()
            .create(true)
            .write(true)
            .open(CSV_FILE)
            .expect("Cannot create CSV file");
        f.write_all(CSV_HEADER.as_bytes())
            .expect("Cannot write CSV header");
    }

    // ── One-time setup: load model outside the loop (mirrors Python) ───────────
    println!("Loading embedding model (once)...");
    let model = embedder::load_model();

    let mut context = String::new();

    // ── 100-iteration profiling loop ───────────────────────────────────────────
    for run in 1..=NUM_RUNS {
        println!("\n{}", "─".repeat(50));
        println!("Run {}/{}", run, NUM_RUNS);

        // 1. PDF read & extract — all files in /pdfs ───────────────────────────
        let t = Instant::now();

        let mut file_content = String::new();
        for pdf_path in &pdf_files {
            let mut doc = PdfDocument::open(pdf_path)
                .unwrap_or_else(|_| panic!("Could not open PDF: {:?}", pdf_path));
            let pages = doc.page_count().expect("Could not get page count");
            for i in 0..pages {
                let page_text = doc.extract_text(i).unwrap_or_default();
                file_content.push_str(&page_text);
                file_content.push_str("\n\n");
            }
        }

        let pdf_read_ms = t.elapsed().as_secs_f64() * 1000.0;
        println!(
            "  PDF Read & Extract:    {:.4} ms  ({} file(s))",
            pdf_read_ms,
            pdf_files.len()
        );

        // 2. Text chunking ─────────────────────────────────────────────────────
        let t = Instant::now();

        let splitter = TextSplitter::new(500);
        let docs: Vec<&str> = splitter.chunks(&file_content).collect();

        let chunking_ms = t.elapsed().as_secs_f64() * 1000.0;
        println!(
            "  Text Chunking:         {:.4} ms  ({} chunks)",
            chunking_ms,
            docs.len()
        );

        // 3. Bulk embedding (model already loaded) ─────────────────────────────
        let t = Instant::now();

        println!("  Encoding chunks in bulk...");
        let embeddings = embedder::embed_batch(&model, docs.clone());

        let embedding_ms = t.elapsed().as_secs_f64() * 1000.0;
        println!("  Model/Embedding Stage: {:.4} ms", embedding_ms);

        // 4. DB insertion (overwrite — store handles reset internally) ──────────
        let t = Instant::now();

        let table = store::get_table().await;
        store::insert(&table, docs.clone(), embeddings).await;

        let row_count = table.count_rows(None).await.unwrap_or(0);
        let db_insert_ms = t.elapsed().as_secs_f64() * 1000.0;
        println!(
            "  DB Init & Insertion:   {:.4} ms  ({} rows)",
            db_insert_ms, row_count
        );

        // 5. Vector search ─────────────────────────────────────────────────────
        let t = Instant::now();

        let query_embedding = embedder::embed(&model, QUESTION);
        let chunks = store::search(&table, query_embedding, 2).await;
        context = chunks.join("\n\n");

        let search_ms = t.elapsed().as_secs_f64() * 1000.0;
        println!("  Search:                {:.4} ms", search_ms);

        // ── Append this run to CSV ─────────────────────────────────────────────
        let mut f = OpenOptions::new()
            .append(true)
            .open(CSV_FILE)
            .expect("Cannot open CSV file for appending");

        writeln!(
            f,
            "{},{:.4},{:.4},{:.4},{:.4},{:.4}",
            run, pdf_read_ms, chunking_ms, embedding_ms, db_insert_ms, search_ms
        )
        .expect("Cannot write CSV row");
    }

    // ── Single LLM call after all profiling runs ───────────────────────────────
    println!("\n{}", "─".repeat(50));
    println!("Running LLM query (single call, not profiled)...");

    let client = Client::new();
    let answer = llm::ask(&client, &api_key, &model_name, &context, QUESTION).await;

    println!("\nAnswer: {}", answer);
    println!("\nProfiling complete. Results saved to '{}'.", CSV_FILE);
}