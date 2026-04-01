use arrow_array::{Array, FixedSizeListArray, RecordBatch, RecordBatchIterator, StringArray};
use arrow_array::types::Float32Type;
use arrow_schema::{DataType, Field, Schema};
use lancedb::{connect, Table};
use lancedb::query::{QueryBase, ExecutableQuery};
use lancedb::table::AddDataMode;
use futures::TryStreamExt;
use std::sync::Arc;

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("text", DataType::Utf8, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                384,
            ),
            false,
        ),
    ]))
}

pub async fn get_table() -> Table {
    let db = connect("data/lancedb").execute().await.unwrap();
    match db.open_table("chunks").execute().await {
        Ok(table) => table,
        Err(_) => db.create_empty_table("chunks", schema()).execute().await.unwrap(),
    }
}

pub async fn insert(table: &Table, texts: Vec<&str>, embeddings: Vec<Vec<f32>>) {
    let texts_array = Arc::new(StringArray::from(texts));

    let embedding_iter = embeddings.into_iter().map(|emb| {
        Some(emb.into_iter().map(Some).collect::<Vec<_>>())
    });

    let embedding_array = Arc::new(
        FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            embedding_iter,
            384,
        ),
    );

    let batch = RecordBatch::try_new(schema(), vec![texts_array, embedding_array]).unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema());

    // Overwrite replaces all existing rows atomically — no drop/recreate needed.
    // This avoids the expensive filesystem deletion that made run 1 an outlier.
    table.add(reader)
        .mode(AddDataMode::Overwrite)
        .execute()
        .await
        .unwrap();
}

pub async fn search(table: &Table, query_embedding: Vec<f32>, limit: usize) -> Vec<String> {
    let results: Vec<RecordBatch> = table
        .query()
        .nearest_to(query_embedding)
        .unwrap()
        .limit(limit)
        .execute()
        .await
        .unwrap()
        .try_collect::<Vec<RecordBatch>>()
        .await
        .unwrap();

    let mut texts = vec![];
    for batch in results {
        let col = batch.column_by_name("text").unwrap();
        let arr = col.as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..arr.len() {
            texts.push(arr.value(i).to_string());
        }
    }
    texts
}