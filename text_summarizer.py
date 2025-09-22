#!/usr/bin/env python3
"""
Map-Reduce Text Summarizer
A Python implementation for summarizing long texts using map-reduce technique with dynamic language support.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Load environment variables
load_dotenv()


class MapReduceSummarizer:
    """Map-Reduce Text Summarizer with dynamic language support"""
    
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        """Initialize the summarizer with Groq LLM"""
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.llm = ChatGroq(api_key=self.api_key, model=model_name)
        self.target_language = "English"
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self._create_prompts()
    
    def _create_prompts(self):
        """Create map and reduce prompts for the current language"""
        self.map_prompt = ChatPromptTemplate.from_template(f"""
Summarize the following text in {self.target_language}. Focus on the key points:

{{text}}

Summary in {self.target_language}:
""")
        
        self.reduce_prompt = ChatPromptTemplate.from_template(f"""
The following is a set of summaries in {self.target_language}:

{{text}}

Take these and distill it into a final, consolidated summary in {self.target_language}. 
The final summary should be comprehensive and capture all the key points.

Final summary in {self.target_language}:
""")
    
    def change_language(self, new_language):
        """Change target language for summarization"""
        self.target_language = new_language
        self._create_prompts()
        return f"Language changed to: {self.target_language}"
    
    def split_text(self, text):
        """Split text into chunks for processing"""
        chunks = self.text_splitter.split_text(text)
        return [Document(page_content=chunk) for chunk in chunks]
    
    def map_reduce_summarize(self, documents):
        """Perform map-reduce summarization on documents"""
        # Step 1: Map - Summarize each chunk
        map_chain = self.map_prompt | self.llm
        chunk_summaries = []
        
        print(f"Processing {len(documents)} chunks...")
        for i, doc in enumerate(documents, 1):
            print(f"Summarizing chunk {i}/{len(documents)}")
            summary = map_chain.invoke({"text": doc.page_content})
            chunk_summaries.append(summary.content)
        
        # Step 2: Reduce - Combine all summaries
        print("Combining summaries...")
        combined_summaries = "\n\n".join(chunk_summaries)
        reduce_chain = self.reduce_prompt | self.llm
        final_summary = reduce_chain.invoke({"text": combined_summaries})
        
        return final_summary.content
    
    def summarize_text(self, text, target_language=None):
        """Main method to summarize text with optional language specification"""
        if target_language and target_language != self.target_language:
            self.change_language(target_language)
        
        print(f"Summarizing text in {self.target_language}...")
        documents = self.split_text(text)
        summary = self.map_reduce_summarize(documents)
        return summary


def main():
    """Main function to demonstrate the summarizer"""
    # Sample text for demonstration
    sample_text = """The computer is something extraordinary.
It's going to probably take over our lives.
That's probably the new industry.
The computer will shape our lives.
It's already doing it quietly ... slowly. We're unaware of it.
We've talked to a great many of these experts, computer experts, for building it. They are not concerned with what happens to the human brain. You understand? They are concerned with creating it. Ah! Not creating it, building it. That's better word.
When the computer takes over ... our lives ... what happens to our brains?
They are better, far quicker, so rapid.
In a second they'll tell you a thousand memories. 
So when they take off, what's going to happen to our brains? Gradually wither? 
Or, be thoroughly employed in amusement ... in entertainment?
Please face all this, for God's sake, this is happening."""
    
    try:
        # Initialize summarizer
        summarizer = MapReduceSummarizer()
        
        # Summarize in English
        print("="*60)
        print("MAP-REDUCE TEXT SUMMARIZER")
        print("="*60)
        
        summary = summarizer.summarize_text(sample_text)
        print(f"\n Summary in English:")
        print("-" * 40)
        print(summary)
        
        # Change language and summarize again
        print("\n" + "="*60)
        summarizer.change_language("French")
        french_summary = summarizer.summarize_text(sample_text)
        print(f"\n Summary in French:")
        print("-" * 40)
        print(french_summary)
        
        # Try another language
        print("\n" + "="*60)
        summarizer.change_language("Spanish")
        spanish_summary = summarizer.summarize_text(sample_text)
        print(f"\n Summary in Spanish:")
        print("-" * 40)
        print(spanish_summary)
        
    except Exception as e:
        print(f" Error: {e}")
        print("💡 Make sure GROQ_API_KEY is set in your .env file")


if __name__ == "__main__":
    main()
