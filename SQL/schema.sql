
CREATE TABLE student_performance(
    id INt PRIMARY KEY AUTO_INCREMENT,
    gender VAECHAR(10),
    race_ethnicity VARCHAR(20),
    parental_education VARCHAR(50),
    lunch VARCHAR(20),
    test_preparation VARCHAR(20),
    math_score INT,
    reading_score INT,
    writing_score INT
);