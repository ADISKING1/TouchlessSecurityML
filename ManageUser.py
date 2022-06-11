import bcrypt

from PasswordUtility import hashPwd, verifyPwd


def createUser(id, name, password):
    import sqlite3
    conn = sqlite3.connect('/home/naditya/Others/Touchless/users.db')
    cursor = conn.cursor()
    print("Opened database successfully")

    mySalt = bcrypt.gensalt()
    password = hashPwd(mySalt, password)
    cursor.execute(
        "INSERT INTO USERS (ID, NAME, PASSWORD) VALUES (?, ?, ?)", (id, name, password))

    conn.commit()
    conn.close()


def getUserNamePassword(id):
    import sqlite3
    conn = sqlite3.connect('/home/naditya/Others/Touchless/users.db')
    cursor = conn.cursor()
    print("Opened database successfully")

    name, password = "", ""

    for row in cursor.execute("SELECT NAME, PASSWORD FROM USERS WHERE ID=" + id):
        name = row[0]
        password = row[1]
        break

    conn.commit()
    conn.close()

    return name, password


# mySalt = bcrypt.gensalt()
# createUser('96', 'Aditya', '1234')

name, password = getUserNamePassword('99')
if(name != ""):
    print(verifyPwd('1234', password))
