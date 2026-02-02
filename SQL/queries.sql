-- 1. Average scores by gender
SELECT gender,
       AVG("math score") AS avg_math,
       AVG("reading score") AS avg_reading,
       AVG("writing score") AS avg_writing
FROM student_performance
GROUP BY gender;


-- 2. Test preparation impact
SELECT "test preparation course" AS test_prep,
       AVG("math score") AS avg_math
FROM student_performance
GROUP BY "test preparation course";


-- 3. Lunch vs performance
SELECT lunch,
       AVG("math score") AS avg_math
FROM student_performance
GROUP BY lunch;


-- 4. Parental education vs reading
SELECT "parental level of education" AS parent_edu,
       AVG("reading score") AS avg_reading
FROM student_performance
GROUP BY "parental level of education";


-- 5. Top 10 students
SELECT gender,
       "race/ethnicity" AS race,
       ("math score"+"reading score"+"writing score")/3 AS avg_score
FROM student_performance
ORDER BY avg_score DESC
LIMIT 10;


-- 6. Students below average in math
SELECT *
FROM student_performance
WHERE "math score" < (
    SELECT AVG("math score")
    FROM student_performance
);
