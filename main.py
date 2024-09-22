import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	group = parser.add_mutually_exclusive_group()
	group.add_argument('-l', '--l_infinity', action='store_true', help='Verify against l_Infinity perturbations')
	group.add_argument('-p', '--patches', action='store_true', help='Verify against patch perturbations')

	args = parser.parse_args()
	print("Hello, test")
