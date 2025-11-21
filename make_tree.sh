#!/bin/zsh

make_tree() {

	# use single-quote ' around patterns with wildcard
	local excludeDirs='coverage|node_modules|.git|.DS_Store'
	local fileName='directory_tree.md'

	# Create the tree output
	# Added -a flag to show hidden files and directories
	tree -a -L 10 -I "$excludeDirs" --dirsfirst -sD --timefmt "%Y-%m-%d" -o "$fileName"

	# Add markdown formatting
	echo "\`\`\`bash" | cat - "$fileName" > temp && mv temp "$fileName"
	echo "\`\`\`" >> "$fileName"

	echo "Directory tree created as $fileName"
}

make_tree
