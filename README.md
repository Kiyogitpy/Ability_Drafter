Dota2 Ability Drafter

This project implements a tool that extracts ability icons from Dota 2 screenshots, matches them against a library using MobileNetV2 features, and displays the results as an overlay. The matching is performed either individually or in batch using cosine similarity between image embeddings. Additionally, the overlay uses PyQt5 and communicates with a Flask game server that handles incoming game state data. Once ingame the overlay will shut off.
Features

    Icon Extraction and Matching:
    Uses MobileNetV2 (pre-trained on ImageNet) to extract features from ability icons, and matches icons via cosine distance.

    Batch Processing and Caching:
    Supports batch processing of images and caching of computed library embeddings for faster subsequent runs.

    Overlay Display:
    Uses PyQt5 to display an overlay with highlighted ability icons.

    Game Server Integration:
    A Flask server receives game state data via POST requests and triggers image cropping and matching accordingly.

Requirements

See the requirements.txt file.
Setup

    Clone the repository:

git clone https://github.com/yourusername/dota2-icon-matcher.git
cd dota2-icon-matcher

Install dependencies:

    pip install -r requirements.txt


Running the Application

You can start the application by running:

python your_main_script.py

This starts the Flask game server and the PyQt5 overlay. When the game state signal is received, the application crops the screenshot, matches the icons with the library, and displays the overlay with the results.
License

This project is licensed under the MIT License.
