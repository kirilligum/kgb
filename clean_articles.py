# import os
import json
from openai import OpenAI, completions
from pathlib import Path
from pydantic import BaseModel


# Initialize OpenAI client
client = OpenAI()


class NewsArticle(BaseModel):
    article_found: bool
    title: str
    publication_date: str
    author: str
    publisher: str
    body_text: str


def clean_article(html_content):
    """Clean article using OpenAI API"""

    extraction_prompt = """
        Objective: Extract the core content of a news article, including the title, publication date, publisher, author, and article body text. The extracted text should be exactly as it appears in the HTML, maintaining punctuation, spacing, and line breaks as they are.

        Requirements:
        1. Variable for Article Detection:
           - Define a variable "article_found".
           - If no valid article text is detected or if the content only includes a paywall message, set "article_found": false. Otherwise, set it to true.

        2. Output as JSON:
           - The extracted data should be output as a JSON object.
                article_found: bool
                title: str
                publication_date: str
                author: str
                publisher: str
                body_text: str
        """

    # extraction_prompt = """
    #     Objective: Extract the core content of a news article, including the title, publication date, publisher, author, and article body text. The extracted text should be exactly as it appears in the HTML, maintaining punctuation, spacing, and line breaks as they are.

    #     Additional Requirements:
    #     1. Variable for Article Detection:
    #        - Define a variable "article_found".
    #        - If no valid article text is detected or if the content only includes a paywall message, set "article_found": false. Otherwise, set it to true.

    #     2. Output as JSON:
    #        - The extracted data should be output as a JSON object.

    #     Steps for Extraction:

    #     1. Identify the HTML Document Structure:
    #        - Focus on the <html>, <head>, and <body> tags.
    #        - The title is generally within a <title> tag in the <head> section, and the article text, date, and author details are within the <body> under <main>, <article>, or nested <div> and <p> tags.

    #        Example (Positive):
    #        <xml>
    #        <html><head><title>Article Title</title></head><body><main><h1>Main Article Heading</h1>...</main></body></html>
    #        </xml>

    #        Example (Negative): Avoid sections like <header>, <footer>, <nav>, and <aside>, which often contain menus, advertisements, or unrelated links.
    #        <xml>
    #        <header><div>Subscribe Now</div></header> <!-- This should be ignored -->
    #        </xml>

    #     2. Extract Title:
    #        - Locate the <title> tag inside the <head>. Capture the entire string within this tag.
    #        - If a <h1> tag is present within the <main> or <article> section, capture this as a secondary title if it appears relevant.

    #        Example (Positive):
    #        <xml>
    #        <title>Elon Musk’s Chances Against OpenAI Look Grim</title>
    #        <main><h1>Elon Musk’s Lawsuit Against OpenAI</h1></main>
    #        </xml>
    #        Extract both "Elon Musk’s Chances Against OpenAI Look Grim" and "Elon Musk’s Lawsuit Against OpenAI".

    #        Example (Negative): Avoid unrelated text such as section headers, which are not part of the article title.
    #        <xml>
    #        <header><h1>Latest News</h1></header> <!-- Ignore this -->
    #        </xml>

    #     3. Extract Publication Date:
    #        - Look for elements containing dates, often found within <time>, <span>, or <div> tags with classes or IDs such as date, timestamp, or published.
    #        - Capture the date text exactly as presented, even if it includes time information.

    #        Example (Positive):
    #        <xml>
    #        <time>October 13, 2024 at 2:00 AM PDT</time>
    #        <div class="date">Published on October 13, 2024</div>
    #        </xml>
    #        Extract "October 13, 2024 at 2:00 AM PDT" or "Published on October 13, 2024".

    #        Example (Negative): Avoid unrelated dates, such as timestamps for comments or recent posts.
    #        <xml>
    #        <span>Last edited: October 12, 2024</span> <!-- Ignore this -->
    #        </xml>

    #     4. Extract Author Information:
    #        - Look for labels such as BY or By, often followed by the author’s name in <span>, <div>, or <a> tags.
    #        - If multiple authors are present, capture all names as listed.

    #        Example (Positive):
    #        <xml>
    #        <span>BY</span><span><a>Christiaan Hetzner</a></span>
    #        <div class="author">By Maria Deutscher</div>
    #        </xml>
    #        Extract "Christiaan Hetzner" and "Maria Deutscher".

    #        Example (Negative): Ignore sections where "BY" does not indicate an author, such as in unrelated text or attributions.
    #        <xml>
    #        <div>BY Popular Demand</div> <!-- Ignore this -->
    #        </xml>

    #     5. Extract Article Body Text:
    #        - Focus on <p>, <div>, or <span> tags within the <main> or <article> sections that appear consecutively and contain the main article text.
    #        - Avoid irrelevant elements like navigation links, advertisements, or social media links by excluding sections labeled menu, footer, header, ad, or similar.

    #        Example (Positive):
    #        <xml>
    #        <p>Elon Musk’s chances of winning his lawsuit against OpenAI look grim...</p>
    #        <p>Musk is seeking compensation...</p>
    #        </xml>
    #        Capture each paragraph as part of the article body text.

    #        Example (Negative): Ignore promotional or unrelated content such as newsletter signups, ads, and social media links.
    #        <xml>
    #        <div>Subscribe to our newsletter for more news.</div> <!-- Ignore this -->
    #        <footer>...</footer> <!-- Ignore this -->
    #        </xml>

    #     6. Retain Formatting:
    #        - Ensure all line breaks, spaces, and punctuation are retained in the output exactly as they appear in the HTML.

    #        Example (Positive):
    #        If the article text appears as:
    #        <xml>
    #        <p>Elon Musk’s chances of winning are slim.</p>
    #        <p>Yet, he remains optimistic.</p>
    #        </xml>
    #        Retain this structure in the output:
    #        Elon Musk’s chances of winning are slim.

    #        Yet, he remains optimistic.

    #     7. Check for Paywall and Article Presence:
    #        - If only a paywall message or placeholder text (e.g., “Subscribe to read more”) is detected without valid article content, set "article_found": false in the output JSON.

    #     8. Optional Sections:
    #        - If the article contains subheadings, such as in <h2> tags, include these in the extraction as part of the body text.

    #        Example (Positive):
    #        <xml>
    #        <h2>The Legal Battle</h2>
    #        <p>Musk’s lawsuit highlights...</p>
    #        </xml>
    #        Capture "The Legal Battle" as a subheading and include it with the article text.

    #     By following this process, the extracted text will focus strictly on the article’s content, omitting unrelated HTML elements and ensuring that the information is captured precisely as it is in the original HTML.
    #     """

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",  # Using standard GPT-4 model
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts clean article text from HTML.",
                },
                {
                    "role": "user",
                    "content": f"{extraction_prompt}\n\nHTML Content:\n{html_content}",
                },
            ],
            response_format=NewsArticle,
        )

        if completion and completion.choices:
            event = completion.choices[0].message.parsed
            if event:
                txt = json.dumps(event.dict(), indent=4)
                return txt

        print("No valid response received from OpenAI API")
        return None

    except Exception as e:
        print(f"Error processing article: {str(e)}")
        return None


def process_directory(input_dir, output_dir):
    """Process all HTML files in directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Process only first 2 articles using slice
    for html_file in list(input_path.glob("*.html")):
        # for html_file in list(input_path.glob("*.html"))[:5]:
        print(f"Processing {html_file.name}")

        # Read HTML content
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Clean article
        cleaned_text = clean_article(html_content)

        if cleaned_text:
            # Save cleaned text
            output_file = output_path / f"{html_file.stem}_cleaned.json"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(cleaned_text)
            print(f"Saved cleaned text to {output_file}")
        else:
            print(f"Failed to clean {html_file.name}")


if __name__ == "__main__":
    input_directory = "data/elon_suing_openai"
    output_directory = "data/cleaned_articles"

    process_directory(input_directory, output_directory)
