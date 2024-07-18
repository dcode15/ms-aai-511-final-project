from feature_extractor import FeatureExtractor

data_dir = "../../data/Composer_Dataset"
composers = ["Bach", "Beethoven", "Chopin", "Mozart"]
scalar_features, multidimensional_features = FeatureExtractor.extract_features_for_multiple_files(data_dir, composers)
scalar_features.head()
