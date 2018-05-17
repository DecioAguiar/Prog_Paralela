declare -a arrb=(8 14 15 16 32)
declare -a arrw=(5 7 9 11 13)

for b in "${arrb[@]}"
do
	for w in "${arrw[@]}"
	do
   		eval "sed -i '5s/.*/#define MASK_WIDTH $w/' smoothresult.cu"
   		eval "sed -i '6s/.*/#define TILE_WIDTH $b/' smoothresult.cu"

   		eval "/usr/local/cuda-8.0/bin/nvcc smoothresult.cu -o smoothresult &> /dev/null"

   		echo "BLOCK $b    WIDTH $w"
   		eval "./smoothresult arq3.ppm"
	done
done
