def to_year_month(datestr):
	'Converts date string in format YYYY-MM-DD to (year, month) tuple'
	return int(datestr[:4]), int(datestr[5:7])

def iter_months(min, max):
	'Iterates year-month tuples from @min to @max'
	cur = min
	while cur <= max:
		yield cur
		if cur[1] == 12:
			cur = cur[0] + 1, 1
		else:
			cur = cur[0], cur[1] + 1
