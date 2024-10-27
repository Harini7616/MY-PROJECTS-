Create  database  ProjectFinal;
use   ProjectFinal;
CREATE TABLE Catalog (
    Catalog_ID INT PRIMARY KEY,
    Name VARCHAR(100) NOT NULL,
    Location VARCHAR(100)
);

CREATE TABLE Genre (
    Genre_ID INT PRIMARY KEY,
    Name VARCHAR(100) NOT NULL,
    Description TEXT
);

CREATE TABLE Author (
    Author_ID INT PRIMARY KEY,
    Name VARCHAR(100) NOT NULL,
    Birth_Date DATE,
    Nationality VARCHAR(50)
);

CREATE TABLE Member (
    Member_ID INT PRIMARY KEY,
    Name VARCHAR(100) NOT NULL,
    Contact_Info VARCHAR(255),
    Join_Date DATE
);

CREATE TABLE Staff (
    Staff_ID INT PRIMARY KEY,
    Name VARCHAR(100) NOT NULL,
    Contact_Info VARCHAR(255),
    Job_Title VARCHAR(100),
    Hire_Date DATE
);

CREATE TABLE Material (
    Material_ID INT PRIMARY KEY,
    Title VARCHAR(255) NOT NULL,
    Publication_Date DATE,
    Catalog_ID INT,
    Genre_ID INT,
    FOREIGN KEY (Catalog_ID) REFERENCES Catalog(Catalog_ID),
    FOREIGN KEY (Genre_ID) REFERENCES Genre(Genre_ID)
);



CREATE TABLE Authorship (
    Authorship_ID INT PRIMARY KEY,
    Author_ID INT,
    Material_ID INT,
    FOREIGN KEY (Author_ID) REFERENCES Author(Author_ID),
    FOREIGN KEY (Material_ID) REFERENCES Material(Material_ID)
);

CREATE TABLE Borrow (
    Borrow_ID INT PRIMARY KEY,
    Material_ID INT,
    Member_ID INT,
    Staff_ID INT,
    Borrow_Date DATE,
    Due_Date DATE,
    Return_Date DATE,
    FOREIGN KEY (Material_ID) REFERENCES Material(Material_ID),
    FOREIGN KEY (Member_ID) REFERENCES Member(Member_ID),
    FOREIGN KEY (Staff_ID) REFERENCES Staff(Staff_ID)
);

INSERT INTO Borrow (Borrow_ID, Material_ID, Member_ID, Staff_ID, Borrow_Date, Due_Date, Return_Date)
VALUES
  (1, 1, 1, 1, '2018-09-12', '2018-10-03', '2018-09-30'),
  (2, 2, 2, 1, '2018-10-15', '2018-11-05', '2018-10-29'),
  (3, 3, 3, 1, '2018-12-20', '2019-01-10', '2019-01-08'),
  (4, 4, 4, 1, '2019-03-11', '2019-04-01', '2019-03-27'),
  (5, 5, 5, 1, '2019-04-20', '2019-05-11', '2019-05-05'),
  (6, 6, 6, 1, '2019-07-05', '2019-07-26', '2019-07-21'),
  (7, 7, 7, 1, '2019-09-10', '2019-10-01', '2019-09-25'),
  (8, 8, 8, 1, '2019-11-08', '2019-11-29', '2019-11-20'),
  (9, 9, 9, 1, '2020-01-15', '2020-02-05', '2020-02-03'),
  (10, 10, 10, 1, '2020-03-12', '2020-04-02', '2020-03-28'),
  (11, 1, 11, 2, '2020-05-14', '2020-06-04', '2020-05-28'),
  (12, 2, 12, 2, '2020-07-21', '2020-08-11', '2020-08-02'),
  (13, 3, 13, 2, '2020-09-25', '2020-10-16', '2020-10-15'),
  (14, 4, 1, 2, '2020-11-08', '2020-11-29', '2020-11-24'),
  (15, 5, 2, 2, '2021-01-03', '2021-01-24', '2021-01-19'),
  (16, 6, 3, 2, '2021-02-18', '2021-03-11', '2021-03-12'),
  (17, 17, 4, 2, '2021-04-27', '2021-05-18', '2021-05-20'),
  (18, 18, 5, 2, '2021-06-13', '2021-07-04', '2021-06-28'),
  (19, 19, 6, 2, '2021-08-15', '2021-09-05', '2021-09-03'),
  (20, 20, 7, 2, '2021-10-21', '2021-11-11', NULL),
  (21, 21, 1, 3, '2021-11-29', '2021-12-20', NULL),
  (22, 22, 2, 3, '2022-01-10', '2022-01-31', '2022-01-25'),
  (23, 23, 3, 3, '2022-02-07', '2022-02-28', '2022-02-23'),
  (24, 24, 4, 3, '2022-03-11', '2022-04-01', '2022-03-28'),
  (25, 25, 5, 3, '2022-04-28', '2022-05-19', '2022-05-18'),
  (26, 26, 6, 3, '2022-06-22', '2022-07-13', '2022-07-08'),
  (27, 27, 7, 3, '2022-08-04', '2022-08-25', '2022-08-23'),
  (28, 28, 8, 3, '2022-09-13', '2022-10-04', '2022-09-28'),
  (29, 29, 9, 3, '2022-10-16', '2022-11-06', '2022-11-05'),
  (30, 30, 8, 3, '2022-11-21', '2022-12-12', '2022-12-05'),
  (31, 1, 9, 4, '2022-12-28', '2023-01-18', NULL),
  (32, 2, 1, 4, '2023-01-23', '2023-02-13', NULL),
  (33, 3, 10, 4, '2023-02-02', '2023-02-23', '2023-02-17'),
  (34, 4, 11, 4, '2023-03-01', '2023-03-22', NULL),
  (35, 5, 12, 4, '2023-03-10', '2023-03-31', NULL),
  (36, 6, 13, 4, '2023-03-15', '2023-04-05', NULL),
  (37, 7, 17, 4, '2023-03-25', '2023-04-15', NULL),
  (38, 8, 8, 4, '2023-03-30', '2023-04-20', NULL),
  (39, 9, 9, 4, '2023-03-26', '2023-04-16', NULL),
  (40, 10, 20, 4, '2023-03-28', '2023-04-18', NULL);
  
  #Query 1
SELECT M.Material_ID, M.Title
FROM Material M
WHERE M.Material_ID NOT IN (
    SELECT B.Material_ID
    FROM Borrow B
    WHERE B.Return_Date IS NULL
);

#Query 2
SELECT M.Title, B.Borrow_Date, B.Due_Date
FROM Material M
INNER JOIN Borrow B ON M.Material_ID = B.Material_ID
WHERE B.Due_Date < '2023-04-01' AND B.Return_Date IS NULL;

#Query 3
SELECT M.Title, COUNT(*) AS BorrowCount
FROM Material M
LEFT JOIN Borrow B ON M.Material_ID = B.Material_ID
WHERE B.Return_Date IS  not NULL
GROUP BY M.Title
ORDER BY BorrowCount DESC
LIMIT 10;

#Query 4
SELECT A.Name AS Author_Name, COUNT(*) AS Material_Count
FROM Author A
INNER JOIN Authorship ASH ON A.Author_ID = ASH.Author_ID
WHERE A.Name = 'Lucas Piki'
GROUP BY A.Name;

# Query 5
SELECT M.Title, COUNT(ASR.Author_ID) AS Author_Count
FROM Material M
INNER JOIN Authorship ASR ON M.Material_ID = ASR.Material_ID
GROUP BY M.Material_ID
HAVING Author_Count >= 2;

#Query 6
SELECT G.Name AS Genre_Name, COUNT(*) AS BorrowCount
FROM Genre G
INNER JOIN Material M ON G.Genre_ID = M.Genre_ID
INNER JOIN Borrow B ON M.Material_ID = B.Material_ID
GROUP BY G.Name
ORDER BY BorrowCount DESC;

#Query 7
SELECT COUNT(*) AS BorrowedMaterials
FROM Borrow
WHERE Borrow_Date BETWEEN '2020-09-01' AND '2020-10-31';

#quey 8
UPDATE Borrow
SET Return_Date = '2023-04-01'
WHERE Material_ID = 1 AND Return_Date IS NULL;
SELECT Material_ID, Return_Date
FROM Borrow
WHERE Material_ID = 1;

-- Step 1: Identify the Member_ID associated with "Emily Miller"
SELECT Member_ID
FROM Member
WHERE Name = 'Emily Miller';

-- Step 2: Delete related records in other tables (e.g., Borrow)
-- First, identify the Borrow records associated with the Member_ID
-- Replace [Emily's Member_ID] with the actual Member_ID retrieved in step 1

DELETE FROM Borrow
WHERE Member_ID = 5;


#Step 3: Delete "Emily Miller" from the Member table
-- Replace [Emily's Member_ID] with the actual Member_ID retrieved in step 1
DELETE FROM Member
WHERE Member_ID = 5;

#Query 10
DELETE FROM Material
WHERE Material_ID = 32;

select * from material;

-- Find the maximum Author_ID in the Author table and increment it by 1
SET @newAuthorID = (SELECT MAX(Author_ID) + 1 FROM Author);

-- Insert the new author "Lucas Luke" using the calculated Author_ID
INSERT INTO Author (Author_ID, Name, Birth_Date, Nationality)
VALUES (@newAuthorID, 'Lucas Luke', NULL, NULL);

-- Find the maximum Material_ID in the Material table and increment it by 1
SET @newMaterialID = (SELECT MAX(Material_ID) + 1 FROM Material);

-- Find the Genre_ID for 'Mystery & Thriller'
SET @genreID = (SELECT Genre_ID FROM Genre WHERE Name = 'Mystery & Thriller');

-- Insert the new material using the calculated Material_ID, Genre_ID, and the newly inserted Author_ID
INSERT INTO Material (Material_ID, Title, Publication_Date, Catalog_ID, Genre_ID)
VALUES (@newMaterialID, 'New book', '2020-08-01', 3, @genreID);


SET @newAuthorshipID = (SELECT MAX(Authorship_ID) + 1 FROM Authorship);

-- Insert the new authorship using the calculated Authorship_ID, Author_ID, and Material_ID
INSERT INTO Authorship (Authorship_ID, Author_ID, Material_ID)
VALUES (@newAuthorshipID, @newAuthorID, @newMaterialID);

SELECT * FROM Material WHERE Title = 'New book';

SELECT a.Name AS Author_Name, m.Title AS Material_Title
FROM Author a
JOIN Authorship auth ON a.Author_ID = auth.Author_ID
JOIN Material m ON auth.Material_ID = m.Material_ID
WHERE a.Name = 'Lucas Luke' AND m.Title = 'New book';

SELECT
    m.member_id, m.member_name, o.material_id, o.due_date
FROM
    members m
JOIN
    overdue_materials o ON m.member_id = o.member_id
WHERE
    o.due_date < CURRENT_DATE;
 UPDATE members
SET
    status = 'inactive'
WHERE
    member_id IN (
        SELECT member_id
        FROM overdue_occurrences
        WHERE occurrence_count >= 3
    );
    
UPDATE members
SET
    status = 'active'
WHERE
    member_id IN (
        SELECT member_id
        FROM overdue_occurrences
        WHERE occurrence_count < 3
    );
