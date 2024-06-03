# capitalize()
my_str = "hello world"
print("capitalize:", my_str.capitalize())  # Output: HELLO WORLD

# casefold()
my_str = "HELLO WORLD"
print("casefold:", my_str.casefold())  # Output: hello world

# center()
my_str = "hello"
print("center:", my_str.center(10, '*'))  # Output: **hello***

# count()
my_str = "hello world"
print("count 'l':", my_str.count('l'))  # Output: 3

# encode()
my_str = "hello world"
print("encode:", my_str.encode())  # Output: b'hello world'

# endswith()
my_str = "hello world"
print("endswith 'world':", my_str.endswith('world'))  # Output: True

# expandtabs()
my_str = "hello\tworld"
print("expandtabs:", my_str.expandtabs(4))  # Output: hello   world

# find()
my_str = "hello world"
print("find 'world':", my_str.find('world'))  # Output: 6

# format()
print("format:", "Hello, {}!".format("world"))  # Output: Hello, world!

# format_map()
my_dict = {"name": "world"}
print("format_map:", "Hello, {name}!".format_map(my_dict))  # Output: Hello, world!

# index()
my_str = "hello world"
print("index 'world':", my_str.index('world'))  # Output: 6

# isalnum()
my_str = "hello123"
print("isalnum:", my_str.isalnum())  # Output: True

# isalpha()
my_str = "hello"
print("isalpha:", my_str.isalpha())  # Output: True

# isdecimal()
my_str = "12345"
print("isdecimal:", my_str.isdecimal())  # Output: True

# isdigit()
my_str = "12345"
print("isdigit:", my_str.isdigit())  # Output: True

# isidentifier()
my_str = "hello_world"
print("isidentifier:", my_str.isidentifier())  # Output: True

# islower()
my_str = "hello"
print("islower:", my_str.islower())  # Output: True

# isnumeric()
my_str = "12345"
print("isnumeric:", my_str.isnumeric())  # Output: True

# isprintable()
my_str = "hello world"
print("isprintable:", my_str.isprintable())  # Output: True

# isspace()
my_str = "   "
print("isspace:", my_str.isspace())  # Output: True

# istitle()
my_str = "Hello World"
print("istitle:", my_str.istitle())  # Output: True

# isupper()
my_str = "HELLO"
print("isupper:", my_str.isupper())  # Output: True

# join()
my_list = ['hello', 'world']
print("join:", ' '.join(my_list))  # Output: hello world

# ljust()
my_str = "hello"
print("ljust:", my_str.ljust(10, '*'))  # Output: hello*****

# lower()
my_str = "HELLO WORLD"
print("lower:", my_str.lower())  # Output: hello world

# lstrip()
my_str = "   hello"
print("lstrip:", my_str.lstrip())  # Output: 'hello'

# maketrans() and translate()
intab = "aeiou"
outtab = "12345"
trantab = str.maketrans(intab, outtab)
my_str = "this is string example"
print("translate:", my_str.translate(trantab))  # Output: th3s 3s str3ng 2x1mpl2

# partition()
my_str = "hello world"
print("partition:", my_str.partition(' '))  # Output: ('hello', ' ', 'world')

# replace()
my_str = "hello world"
print("replace:", my_str.replace("world", "Python"))  # Output: hello Python

# rfind()
my_str = "hello world hello"
print("rfind 'hello':", my_str.rfind('hello'))  # Output: 12

# rindex()
my_str = "hello world hello"
print("rindex 'hello':", my_str.rindex('hello'))  # Output: 12

# rjust()
my_str = "hello"
print("rjust:", my_str.rjust(10, '*'))  # Output: *****hello

# rpartition()
my_str = "hello world"
print("rpartition:", my_str.rpartition(' '))  # Output: ('hello', ' ', 'world')

# rsplit()
my_str = "hello world"
print("rsplit:", my_str.rsplit(' '))  # Output: ['hello', 'world']

# rstrip()
my_str = "hello   "
print("rstrip:", my_str.rstrip())  # Output: 'hello'

# split()
my_str = "hello world"
print("split:", my_str.split(' '))  # Output: ['hello', 'world']

# splitlines()
my_str = "hello\nworld"
print("splitlines:", my_str.splitlines())  # Output: ['hello', 'world']

# startswith()
my_str = "hello world"
print("startswith 'hello':", my_str.startswith('hello'))  # Output: True

# strip()
my_str = "   hello   "
print("strip:", my_str.strip())  # Output: 'hello'

# swapcase()
my_str = "Hello World"
print("swapcase:", my_str.swapcase())  # Output: hELLO wORLD

# title()
my_str = "hello world"
print("title:", my_str.title())  # Output: Hello World

# upper()
my_str = "hello world"
print("upper:", my_str.upper())  # Output: HELLO WORLD

# zfill()
my_str = "42"
print("zfill:", my_str.zfill(5))  # Output: 00042
