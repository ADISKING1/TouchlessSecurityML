import bcrypt

from PasswordUtility import hashPwd, verifyPwd


def createUser(id,  name, password):
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


def getUserPassword(id):
    import sqlite3
    conn = sqlite3.connect('/home/naditya/Others/Touchless/users.db')
    cursor = conn.cursor()
    print("Opened database successfully")

    for row in cursor.execute("SELECT PASSWORD FROM USERS WHERE ID=" + id):
        password = row[0]
        break

    conn.commit()
    conn.close()

    return password


mySalt = bcrypt.gensalt()
createUser('96', 'Aditya', '1234')

password = getUserPassword('96')
print(verifyPwd('1234', password))
