Core Principles of OOP
Classes and Objects
Encapsulation
Inheritance
Polymorphism
Abstraction

1. Classes and Objects
Class: A blueprint for creating objects. It defines a set of attributes and methods that the objects created from the class will have.
Object: An instance of a class.
python
Copy code
# Define a class
class Car:
    # Constructor
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
    
    # Method to display car details
    def display_info(self):
        print(f"{self.year} {self.make} {self.model}")

# Create an object (instance) of the Car class
my_car = Car("Toyota", "Corolla", 2020)

# Access attributes and methods
print(my_car.make)  # Output: Toyota
my_car.display_info()  # Output: 2020 Toyota Corolla
2. Encapsulation
Encapsulation: The bundling of data (attributes) and methods that operate on the data into a single unit or class. It also involves restricting access to certain components.
python
Copy code
class Car:
    def __init__(self, make, model, year):
        self._make = make  # Protected attribute
        self.__model = model  # Private attribute
        self.year = year
    
    def display_info(self):
        print(f"{self.year} {self._make} {self.__model}")
    
    # Getter for the private attribute
    def get_model(self):
        return self.__model

# Create an object of the Car class
my_car = Car("Toyota", "Corolla", 2020)

# Accessing protected and private attributes (not recommended)
print(my_car._make)  # Output: Toyota
# print(my_car.__model)  # Raises AttributeError

# Using getter method
print(my_car.get_model())  # Output: Corolla
3. Inheritance
Inheritance: A mechanism by which one class (child class) can inherit attributes and methods from another class (parent class). It promotes code reuse.
python
Copy code
# Parent class
class Vehicle:
    def __init__(self, make, model):
        self.make = make
        self.model = model
    
    def display_info(self):
        print(f"Vehicle: {self.make} {self.model}")

# Child class
class Car(Vehicle):
    def __init__(self, make, model, year):
        super().__init__(make, model)
        self.year = year
    
    # Method overriding
    def display_info(self):
        print(f"Car: {self.year} {self.make} {self.model}")

# Create an object of the Car class
my_car = Car("Toyota", "Corolla", 2020)
my_car.display_info()  # Output: Car: 2020 Toyota Corolla
4. Polymorphism
Polymorphism: The ability to use a single interface to represent different underlying data types. In Python, it allows methods to do different things based on the object it is acting upon.
python
Copy code
class Vehicle:
    def display_info(self):
        print("This is a vehicle")

class Car(Vehicle):
    def display_info(self):
        print("This is a car")

class Motorcycle(Vehicle):
    def display_info(self):
        print("This is a motorcycle")

# Function demonstrating polymorphism
def show_vehicle_info(vehicle):
    vehicle.display_info()

# Create objects
car = Car()
motorcycle = Motorcycle()

# Pass objects to the function
show_vehicle_info(car)  # Output: This is a car
show_vehicle_info(motorcycle)  # Output: This is a motorcycle
5. Abstraction
Abstraction: The concept of hiding the complex implementation details and showing only the necessary features of an object. In Python, abstraction is often achieved through abstract classes and interfaces (using the abc module).
python
Copy code
from abc import ABC, abstractmethod

# Abstract class
class Animal(ABC):
    @abstractmethod
    def make_sound(self):
        pass

# Dog is a concreate class of Animal, so it has to implement the abstractmethod make_sound
class Dog(Animal):
    def make_sound(self):
        print("Bark")

# Cat is a concreate class of Animal, so it has to implement the abstractmethod make_sound
class Cat(Animal):
    def make_sound(self):
        print("Meow")

# Create objects
dog = Dog()
cat = Cat()

# Call abstract method
dog.make_sound()  # Output: Bark
cat.make_sound()  # Output: Meow
Summary
Classes and Objects: Define and instantiate reusable blueprints.
Encapsulation: Bundle data and methods, restrict access.
Inheritance: Promote code reuse by inheriting attributes and methods.
Polymorphism: Enable methods to process objects differently based on their class.
Abstraction: Hide complexity by using abstract classes and methods
