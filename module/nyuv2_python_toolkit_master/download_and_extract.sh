wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
wget https://inf.ethz.ch/personal/ladickyl/nyu_normals_gt.zip

python extract_nyuv2.py --mat nyu_depth_v2_labeled.mat --normal_zip nyu_normals_gt.zip  --data_root NYUv2 --save_colored