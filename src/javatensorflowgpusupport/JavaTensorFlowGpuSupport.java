/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package javatensorflowgpusupport;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
/**
 *
 * @author user
 */
public class JavaTensorFlowGpuSupport {

    /**
     * @param args the command line arguments
     * @throws java.lang.Exception
     */
  public static void main(String[] args) throws Exception {
    try (Graph g = new Graph()) {
      final String value = "Hello from " + TensorFlow.version();

      // Construct the computation graph with a single operation, a constant
      // named "MyConst" with a value "value".
      try (Tensor t = Tensor.create(value.getBytes("UTF-8"))) {
        // The Java API doesn't yet include convenience functions for adding operations.
        g.opBuilder("Const", "MyConst").setAttr("dtype", t.dataType()).setAttr("value", t).build();
      }

      // Execute the "MyConst" operation in a Session.
      try (Session s = new Session(g);
           Tensor output = s.runner().fetch("MyConst").run().get(0)) {
        System.out.println(new String(output.bytesValue(), "UTF-8"));
      }
    }
  }
    
}
