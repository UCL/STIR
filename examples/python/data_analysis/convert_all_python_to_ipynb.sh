
# Converts all scripts in the directory to ipynb format for Jupyter Notebooks

for python_file in *".py"; do 
	p2j ${python_file}
done
