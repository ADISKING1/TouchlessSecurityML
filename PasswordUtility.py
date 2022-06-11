import bcrypt


def hashPwd(mySalt, pwd):
    bytePwd = pwd.encode('utf-8')
    return bcrypt.hashpw(bytePwd, mySalt)


def verifyPwd(pwd, hashedPwd):
    bytePwd = pwd.encode('utf-8')
    return bcrypt.checkpw(bytePwd, hashedPwd)


# pwd = '1234'
#
# mySalt = bcrypt.gensalt()
# hashedPassword = hashPwd(mySalt, pwd)
# print(mySalt, hashedPassword)
#
# print(verifyPwd(pwd, hashedPassword))
# # Output: True
#
