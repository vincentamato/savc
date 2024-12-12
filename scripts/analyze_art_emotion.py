import os
import json
import base64
import asyncio
import aiofiles
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set
from tqdm.asyncio import tqdm
import aiohttp
import tempfile
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

SAVE_FREQUENCY = 100  # Save every N images
CHUNK_SIZE = 5000    # Process images in chunks to manage memory
MAX_RETRIES = 3      # Number of retries for failed requests
RETRY_DELAY = 1      # Delay between retries in seconds


PROMPT = """Analyze this artwork's emotional resonance and expressive qualities. 

Choose ONE primary emotion from this carefully curated set that best captures the artwork's dominant feeling:

- Transcendent (spiritual, ethereal, sublime)
- Melancholic (wistful, longing, contemplative)
- Turbulent (dramatic, intense, passionate)
- Euphoric (ecstatic, joyful, celebratory)
- Somber (serious, grave, dignified)
- Intimate (tender, gentle, nurturing)
- Dynamic (energetic, forceful, vigorous)
- Mysterious (enigmatic, dreamlike, uncertain)
- Serene (peaceful, harmonious, balanced)
- Unsettling (disquieting, anxious, tense)

Also provide:
1. Emotional Intensity (1-5): How strongly does the artwork convey this emotion?
2. Psychological Depth (1-5): How complex and layered is the emotional content?

Respond ONLY with these three values in exactly this format:
emotion,intensity,depth"""


class ParallelArtEmotionAnalyzer:
    def __init__(self, api_key: str, max_concurrent: int = 5):
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        self.results: Dict = {}
        self.failed_images: List = []
        self.sem = asyncio.Semaphore(max_concurrent)
        self.thread_pool = ThreadPoolExecutor(max_workers=min(32, max_concurrent))
        self.save_counter = 0
        
    async def load_results(self, save_path: str) -> None:
        """Load existing results from file"""
        if os.path.exists(save_path):
            try:
                async with aiofiles.open(save_path, 'r') as f:
                    content = await f.read()
                    data = json.loads(content)
                    self.results = data.get('results', {})
            except json.JSONDecodeError:
                print("Error loading existing results - starting fresh")
                self.results = {}

    async def encode_image_async(self, image_path: str) -> bytes:
        """Read image bytes asynchronously using thread pool"""
        return await asyncio.get_event_loop().run_in_executor(
            self.thread_pool,
            self.encode_image,
            image_path
        )

    def encode_image(self, image_path: str) -> bytes:
        """Read image bytes"""
        with open(image_path, "rb") as image_file:
            return image_file.read()

    async def analyze_single_image(self, image_path: str) -> Dict:
        """Analyze a single image with retries"""
        async with self.sem:
            for attempt in range(MAX_RETRIES):
                try:
                    # Load image asynchronously
                    image_data = await self.encode_image_async(image_path)
                    
                    # Create Gemini content parts
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.model.generate_content([
                            PROMPT,
                            {"mime_type": "image/jpeg", "data": image_data}
                        ])
                    )
                    
                    response_text = response.text.strip()
                    parts = [p.strip() for p in response_text.split(',')]
                    
                    if len(parts) >= 3:
                        result = {
                            'path': image_path,
                            'emotion': parts[0].lower(),
                            'intensity': float(parts[1]),
                            'depth': float(parts[2]),
                            'timestamp': datetime.now().isoformat()
                        }
                        self.results[image_path] = result
                        
                        # Increment save counter and save if needed
                        self.save_counter += 1
                        if self.save_counter >= SAVE_FREQUENCY:
                            await self.save_results_async(self.current_save_path)
                            self.save_counter = 0
                            
                        return result
                    
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY)
                        continue
                    error_result = {
                        'path': image_path,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    self.failed_images.append(error_result)
                    return error_result

        return None

    async def save_results_async(self, save_path: str):
        """Save results atomically using a temporary file"""
        # Create the output data
        output_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_images': len(self.results),
                'failed_images': len(self.failed_images)
            },
            'results': self.results
        }
        
        # Create a temporary file in the same directory as the target file
        save_dir = os.path.dirname(save_path)
        with tempfile.NamedTemporaryFile(mode='w', dir=save_dir, delete=False) as temp_file:
            # Write to the temporary file
            json.dump(output_data, temp_file, indent=2)
            temp_path = temp_file.name

        try:
            # Rename the temporary file to the target file (atomic operation)
            os.replace(temp_path, save_path)
        except Exception as e:
            # If something goes wrong, clean up the temporary file
            os.unlink(temp_path)
            raise e

    async def process_chunk(self, chunk: List[str], save_path: str, pbar: tqdm) -> None:
        """Process a chunk of images"""
        tasks = [self.analyze_single_image(path) for path in chunk]
        for completed_task in asyncio.as_completed(tasks):
            result = await completed_task
            if result:
                pbar.update(1)

    async def analyze_directory(self, directory_path: str, save_path: str):
        """Analyze all images in a directory with chunked parallel processing"""
        self.current_save_path = save_path
        await self.load_results(save_path)
        
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(Path(directory_path).rglob(f'*{ext}'))
        image_files = [str(p) for p in image_files]
        
        # Filter out already processed images
        remaining_images = [img for img in image_files if img not in self.results]
        
        print(f"Found {len(image_files)} total images")
        print(f"Already processed: {len(self.results)} images")
        print(f"Remaining to process: {len(remaining_images)} images")
        
        # Process in chunks with progress bar
        with tqdm(total=len(remaining_images), desc="Processing images") as pbar:
            for i in range(0, len(remaining_images), CHUNK_SIZE):
                chunk = remaining_images[i:i + CHUNK_SIZE]
                await self.process_chunk(chunk, save_path, pbar)
                # Force save after each chunk
                await self.save_results_async(save_path)
        
        # Save failed images if any
        if self.failed_images:
            failed_path = save_path.replace('.json', '_failed.json')
            async with aiofiles.open(failed_path, 'w') as f:
                await f.write(json.dumps(self.failed_images, indent=2))
            print(f"\nFailed images saved to: {failed_path}")
        
        print(f"\nAnalysis complete:")
        print(f"Successfully processed: {len(self.results)} images")
        print(f"Failed: {len(self.failed_images)} images")

async def main():
    # Configuration
    API_KEY = os.environ.get('GOOGLE_API_KEY')
    INPUT_DIR = "data/artbench"
    OUTPUT_FILE = "data/art_emotions.json"
    MAX_CONCURRENT = 1000  # Adjust based on API rate limits
    
    # Initialize analyzer
    analyzer = ParallelArtEmotionAnalyzer(
        api_key=API_KEY,
        max_concurrent=MAX_CONCURRENT
    )
    
    # Run analysis
    await analyzer.analyze_directory(INPUT_DIR, OUTPUT_FILE)

if __name__ == "__main__":
    asyncio.run(main())