from tkinter import *
import tkinter.font as font


def onSubmit():
    global output1, output2, output3, output4, output5, output6, output7, output8
    output1 = entry1.get()
    output2 = entry2.get()
    output3 = entry3.get()
    output4 = entry4.get()
    output5 = entry5.get()
    output6 = entry6.get()
    output7 = entry7.get()
    output8 = entry8.get()
    # root.destroy()
    root.quit()


def widget_input():

    label0 = Label(root, text="Bilde nun die Pluralform folgender Wörter. \n\nSobald du alles eingetragen hast,"
                              "\nklicke auf 'OK' und das Experiment ist dann zu Ende!  ", font=("Helvetica bold", 20))
    label1 = Label(root, text="talo (Haus)           --> ?", font=("Helvetica bold", 25))
    label2 = Label(root, text="tie (Weg)             --> ?", font=("Helvetica bold", 25))
    label3 = Label(root, text="kirja (Buch)         --> ?", font=("Helvetica bold", 25))
    label4 = Label(root, text="katu (Straße)       --> ?", font=("Helvetica bold", 25))
    label5 = Label(root, text="jalka (Fuß)          --> ?", font=("Helvetica bold", 25))
    label6 = Label(root, text="sopu (Harmonie) --> ?", font=("Helvetica bold", 25))
    label7 = Label(root, text="sota (Krieg)         --> ?", font=("Helvetica bold", 25))
    label8 = Label(root, text="tikka (Dart)         --> ?", font=("Helvetica bold", 25))

    global entry1, entry2, entry3, entry4, entry5, entry6, entry7, entry7, entry8
    entry1 = Entry(root, font="Helvetica 30 bold", width=30)
    entry2 = Entry(root, font="Helvetica 30 bold", width=30)
    entry3 = Entry(root, font="Helvetica 30 bold", width=30)
    entry4 = Entry(root, font="Helvetica 30 bold", width=30)
    entry5 = Entry(root, font="Helvetica 30 bold", width=30)
    entry6 = Entry(root, font="Helvetica 30 bold", width=30)
    entry7 = Entry(root, font="Helvetica 30 bold", width=30)
    entry8 = Entry(root, font="Helvetica 30 bold", width=30)

    label0.grid(row=1, column=1, padx=10, pady=10)

    label1.grid(row=3, column=0, padx=10, pady=10)
    entry1.grid(row=3, column=1, sticky="ew")

    label2.grid(row=5, column=0, padx=10, pady=10)
    entry2.grid(row=5, column=1, sticky="ew")

    label3.grid(row=7, column=0, padx=10, pady=10)
    entry3.grid(row=7, column=1, sticky="ew")

    label4.grid(row=9, column=0, padx=10, pady=10)
    entry4.grid(row=9, column=1, sticky="ew")

    label5.grid(row=11, column=0, padx=10, pady=10)
    entry5.grid(row=11, column=1, sticky="ew")

    label6.grid(row=13, column=0, padx=10, pady=10)
    entry6.grid(row=13, column=1, sticky="ew")

    label7.grid(row=15, column=0, padx=10, pady=10)
    entry7.grid(row=15, column=1, sticky="ew")

    label8.grid(row=17, column=0, padx=10, pady=10)
    entry8.grid(row=17, column=1, sticky="ew")

    # Command tells the form what to do when the button is clicked
    btn = Button(root, text="OK", command=onSubmit, bg='#bd0031')
    btn['font'] = font.Font(family='Helvetica', size=30)
    btn.grid(row=22, column=1,  padx=20, pady=20, sticky="ew")


def main():
    global root
    root = Tk()
    root.geometry("1200x1000")
    widget_input()
    root.mainloop()
    user_input = (output1, output2, output3, output4, output5, output6, output7, output8)
    root.destroy()

    return user_input
