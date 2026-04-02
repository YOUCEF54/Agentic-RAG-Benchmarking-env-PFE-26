use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

pub fn load_model() -> TextEmbedding {
    TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::BGESmallENV15)
    ).unwrap()
}

pub fn embed(model: &TextEmbedding, text: &str) -> Vec<f32> {
    let results = model.embed(vec![text], None).unwrap();
    results[0].clone()
}

pub fn embed_batch(model: &TextEmbedding, text_list: Vec<&str>) -> Vec<Vec<f32>>{
   model.embed(text_list, None).unwrap() 
}
