git add --all .
IF "%1"=="" (
	git commit
) ELSE (
	IF "%1"=="f" (
		git commit -m "minor fix"
	) ELSE (
		git commit -m "%1"
	)
)
git push origin master