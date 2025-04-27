# We define all the global variables here, so that they are easily accessible from every module

# placeholders
map = None    # Map with 3d points based on ORB features
cgraph = None # Convisibility graph
vocab = None  # Basically contains 1000 descriptors
bow_db = None # contains visual_word_id -> keyframe_that_sees_it dicts
