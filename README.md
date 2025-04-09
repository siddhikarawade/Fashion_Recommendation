# Fashion Recommender App

This Flask web application processes a raw clothing dataset using a YOLOv8 model, segregates images into three categories (upper, lower, and full), and logs approved images into a CSV file. If an image contains both an upper and a lower detection, it is copied for Siamese training. A Siamese network (optional) is available for training, and the recommendation engine uses cosine similarity and complementary color matching to suggest outfits. The frontend is styled in a Pinterest-like grid similar to Amazon/Myntra/AJIO.

## Folder Structure

fashion_app/
├── app.py                         # Main Flask app: handles routes for home, admin, upload, camera, recommendations
├── preprocess.py                  # Preprocessing script: processes raw images from preprocess_input, segregates into upper, lower, full, creates siamese pairs, logs CSV
├── train_siamese.py               # (Optional) Siamese network training script using images from static/siamese_train
├── recommendation_engine.py       # Recommendation module: extracts features, computes cosine similarity, returns complementary items
├── requirements.txt               # Python package dependencies
├── README.md                      # Setup, deployment, and usage instructions
├── preprocess_input/              # Raw dataset folder (e.g., with train, valid, test, and data.yaml)
├── static/
│   ├── preprocessed/              # Organized images after preprocessing
│   │   ├── upper/                # Upper garments saved as <class>.jpg (e.g., T-shirts.jpg)
│   │   ├── lower/                # Lower garments saved as <class>.jpg (e.g., Denim-pants.jpg)
│   │   └── full/                 # Full outfits saved with numeric indices (e.g., one_piece1.jpg or men_1.jpg)
│   ├── siamese_train/             # Images with both upper and lower detections for Siamese network training
│   ├── uploads/                   # User-uploaded or captured images
│   └── approved_images.csv        # CSV log with metadata of approved images
└── templates/                     # HTML templates
    ├── index.html                 # Home page (upload & capture options)
    ├── results.html               # Recommendations displayed in a Pinterest-style grid
    ├── camera.html                # Camera capture page
    └── admin.html                 # Admin panel for triggering preprocessing

