"""Simple test without import issues"""
def test_huggingface():
    try:
        # Test basic imports
        import transformers
        import datasets
        print(f"Transformers version: {transformers.__version__}")
        print(f"Datasets version: {datasets.__version__}")
        
        # Test loading a tiny dataset
        from datasets import load_dataset
        print("Loading small dataset sample...")
        dataset = load_dataset("imdb", split="train[:10]")
        print(f"Loaded {len(dataset)} examples")
        print("Sample text:", dataset[0]['text'][:100] + "...")
        
        print("Hugging Face integration test PASSED!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_huggingface()
