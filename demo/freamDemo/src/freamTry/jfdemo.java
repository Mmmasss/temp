package freamTry;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.net.NoRouteToHostException;

public class jfdemo {
    public static void main(String[] args)
    {
        new Calculator().showFrame();
    }



}
class Calculator extends JFrame
{
    JTextField num1 = new JTextField(5);
    JTextField num2 = new JTextField(5);
    JTextField num3 = new JTextField(7);
    JLabel label = new JLabel("+");
    JButton button = new JButton("=");
    public void showFrame()
    {
        //component:3textField,label,button


        CalcuListener buttonListen = new CalcuListener(this);
        button.addActionListener(buttonListen);
        setLayout(new FlowLayout());
        add(num1);
        add(label);
        add(num2);
        add(button);
        add(num3);
        pack();
        setVisible(true);
    }

}

class CalcuListener implements ActionListener
{

    Calculator calculator = null;
    public CalcuListener(Calculator calcu)
    {
        calculator=calcu;
    }
    @Override
    public void actionPerformed(ActionEvent e) {
        int n1=Integer.parseInt(calculator.num1.getText());
        int n2=Integer.parseInt(calculator.num2.getText());
        calculator.num3.setText(""+(n1+n2));
        calculator.num2.setText("");
        calculator.num1.setText("");
    }
}


