arch=$(find ./results/ -maxdepth 1 -type d | tail -n +2)

for item in $arch; do
	result="${item##*/}"
	printf "%s\n" $result
	head -19 $item/predictions.json | tail -5 | sed -e 's/^[ \t]*//'
	printf "\n"
done
